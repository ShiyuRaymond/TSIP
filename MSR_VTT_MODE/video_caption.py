import torch
import torch.nn as nn
import torch.nn.functional as F


from mmdet.apis import init_detector
from mmengine import Config
from modules.cross_attention_fusion import CrossAttentionFusion, AsymmetricFusion
from modules.llm_captioner import VisualSoftPromptEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class YOLOWorldCaptioning(nn.Module):
    def __init__(self, 
                 config,
                 
                 ):
        super().__init__()
       
         # 1. 加载 YOLO-WORLD 模型（含 backbone、neck、bbox_head）
        cfg = Config.fromfile(config.yolo_config_path)
        cfg = Config.fromfile(config.yolo_config_path)
        cfg.work_dir = config.yolo_work_dir
        self.detector = init_detector(cfg, checkpoint=config.yolo_checkpoint)
       
        # 2. 加载视频 backbone（注意该模块内部一般不会传入梯度计算，因此可用 no_grad）
        
        encoder_layer = TransformerEncoderLayer(
            d_model=config.projector_dim,
            nhead=8,
            dim_feedforward=config.projector_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.projector = nn.Sequential(
            nn.Linear(config.original_dim, config.projector_dim),
            TransformerEncoder(encoder_layer, num_layers=2),
        )
        
        # 3. 融合模块（设为可训练）
        self.fusion_module = CrossAttentionFusion(query_dim=config.visual_feat_dim, context_dim=config.visual_feat_dim, num_heads=8, num_layers=7)

        # 4. 视频 caption 模型，用于接收 soft prompt（融合后的 embedding）计算 loss 或生成 caption
        
        # 视觉特征到LLM embedding空间的投影层 (可训练)
        self.llm_embed_dim = config.hidden_size
        # soft prompt长度 (query数量)
        self.soft_prompt_len = config.soft_prompt_len

        self.vis_proj = QFormerPromptEncoder(
            visual_dim=config.visual_feat_dim,     # e.g. 512
            prompt_dim=self.llm_embed_dim,        # e.g. 768
            num_query=self.soft_prompt_len,   # 你要生成的 soft prompt 长度
        )
        
    def forward(self, frames_images, video_embedding):
        """
        参数说明：
          frames_images: Tensor，形状 [B, C, T, H, W]（视频帧预先处理成批次输入）
          video_embedding: Tensor， shape:[B, T, C]
          captions: 可选 list，长度为 B，用于训练时计算 loss
        返回：
          如果 captions 不为 None，则返回 caption 模型计算的 loss；
          否则返回生成的 caption 文本列表。
        """
        frames_images = frames_images.permute(0,2,1,3,4).contiguous() #[B,T,C,H,W]
        B, num_frames = frames_images.shape[:2]

        # 1. 计算视频 embedding
        video_embedding = self.projector(video_embedding)

        # 2. 获取 YOLO-WORLD backbone 的图像特征
        # 将帧图片展平为 [B * num_frames, 3, H, W]
        frames_images_flat = frames_images.view(-1, *frames_images.shape[2:])
       
        with torch.no_grad():
            img_feats_raw = self.detector.backbone.forward_image(frames_images_flat)
    
        # 将每个尺度特征恢复成 [B, num_frames, ...]
        img_feats = [feat.view(B, num_frames, *feat.shape[1:]) for feat in img_feats_raw]
       
        # 重组为列表，每个元素对应一个样本，包含该样本所有帧的特征（列表长度为 num_frames）
        img_feats = [list(x) for x in zip(*img_feats)]

        # 3. 利用 detector neck 模块将图像特征和视频 embedding 进行融合
        neck_feats = []
        for sample_feats, v in zip(img_feats, video_embedding):
            # v 的 shape 为 [num_frames, visual_feat_dim]，扩展后作为辅助信息送入 neck 模块
            neck_feats.append(self.detector.neck(sample_feats, v.unsqueeze(1).expand(-1, num_frames, -1)))
        # 假定 neck 模块的输出为列表：前几项为 image 特征，最后一项为 video 特征
        video_feats_yolo = [nf[-1] for nf in neck_feats]
        
       
        # video_feats_mae 做一次时序融合
        
        # B,n_frame, sequence_len, D -> B,sequence_len, D
        yolo_featus_polles = torch.stack(video_feats_yolo).mean(dim=1)
        # 融合 YOLO 和 VideoMAE 特征
        # video_feats_yolos = torch.stack(video_feats_yolo).transpose(1,2).mean(dim=1)
        fused_embeddings = self.fusion_module(yolo_featus_polles, yolo_featus_polles)
        # fused_embeddings= self.global_local_fusion(yolo_featus_mae_polles, video_feats_mae_pooled,)
        soft_prompt = self.vis_proj(fused_embeddings)
        # soft_prompt = self.soft_prompt_transformer(soft_prompt)
        # print('soft shape is ', soft_prompt.shape)
        del frames_images_flat, img_feats_raw
        torch.cuda.empty_cache()
        return fused_embeddings, soft_prompt




class QFormerPromptEncoder(nn.Module):
    def __init__(self,
                 visual_dim: int,       # 输入视觉特征维度，如 512
                 prompt_dim: int,       # 输出 soft prompt 的维度（即 LLM 的 embedding dim）
                 num_query: int = 32,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 visual_len: int = 32):
        """
        参数：
        - visual_dim: 输入视觉特征的维度（来自 fused_emd）
        - prompt_dim: LLM 所期望的 embedding dim（如 OPT 是 2048）
        - num_query: learnable query token 个数
        - visual_len: 视觉特征固定长度（帧数或 patch 数）
        """
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_query = num_query
        self.visual_len = visual_len

        # 可学习 query tokens：作为 soft prompt 的基础
        self.query_tokens = nn.Parameter(torch.randn(1, num_query, prompt_dim))

        # 映射视觉特征到 prompt_dim
        self.vision_proj = nn.Linear(visual_dim, prompt_dim)

        # 可学习的视觉位置编码（帧位置 or patch 位置）
        self.visual_pos = nn.Parameter(torch.randn(1, visual_len, prompt_dim))

        # Q-Former 结构：Decoder + Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=prompt_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True  # 推荐
        )
        self.cross_attn = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 输出做一层 LayerNorm（增强稳定性）
        # self.output_norm = nn.LayerNorm(prompt_dim)

    def forward(self, visual_feat: torch.Tensor):
        """
        输入：
        - visual_feat: [B, F, Dv] → 视频帧或空间 patch 表征
        输出：
        - soft_prompt: [B, Nq, D] → 给 LLM 使用的 prompt
        """
        B, F, _ = visual_feat.shape
        assert F == self.visual_len, f"当前 QFormer 仅支持固定帧数输入 F={self.visual_len}"

        # 投影视觉特征维度 + 位置编码
        visual_proj = self.vision_proj(visual_feat)                     # [B, F, D]
        visual_proj = visual_proj + self.visual_pos                   # [B, F, D]

        # 准备 query tokens
        query = self.query_tokens.expand(B, -1, -1)                     # [B, Nq, D]

        # Cross-attention：Query attend to visual features
        output = self.cross_attn(tgt=query, memory=visual_proj)        # [B, Nq, D]
        return output
        # return self.output_norm(output)


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: [B, num_tokens, C], num_tokens可以是patch数目或其他空间维度
        Returns:
            enhanced_feats: [B, num_tokens, C]
        """
        enhanced_feats = self.encoder(x)
        return enhanced_feats
    
class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 可学习的时序位置编码 (optional)
        self.positional_encoding = nn.Parameter(torch.randn(1, 32, embed_dim))  # 假设最大帧数100
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            temporal_feats: [B, T, C]
        """
        B, T, C = x.shape
        pos_emb = self.positional_encoding[:, :T, :]
        x = x + pos_emb  # 加入时序位置编码
        temporal_feats = self.encoder(x)
        return temporal_feats
    
def uniform_sample_frames(video_tensor, num_out_frames=16):
    """
    输入：
        video_tensor: [B, num_frames, 3, H, W]
        num_out_frames: 目标帧数
    输出：
        sampled_tensor: [B, num_out_frames, 3, H, W]
    """
    B, num_frames, C, H, W = video_tensor.shape
    # 采样下标
    idxs = torch.linspace(0, num_frames - 1, steps=num_out_frames).long()
    # 支持 batch 采样
    sampled = video_tensor[:, idxs, :, :, :]   # [B, num_out_frames, 3, H, W]
    return sampled


class HierarchicalFusionTransformer(nn.Module):
    def __init__(self, global_dim, local_dim, hidden_dim=512, n_heads=8):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, hidden_dim)
        self.local_proj = nn.Linear(local_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, global_feat, local_feats):
        global_emb = self.global_proj(global_feat) # [B,T1,D]
        T1 = global_emb.shape[1]
        local_emb = self.local_proj(local_feats) # [B,T2,D]
        combined = torch.cat([global_emb, local_emb], dim=1)
        fused = self.transformer(combined)
        return fused[:, :T1, :] # 用 global 位置表示

