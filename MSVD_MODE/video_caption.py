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
        # self.projector = nn.Sequential(nn.LayerNorm(config.original_dim),
        #                            nn.Linear(config.original_dim, config.projector_dim))
        
        encoder_layer = TransformerEncoderLayer(
            d_model=config.projector_dim,
            nhead=4,
            dim_feedforward=config.projector_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.projector = nn.Sequential(
            # nn.LayerNorm(config.original_dim),
            nn.Linear(config.original_dim, config.projector_dim),
            TransformerEncoder(encoder_layer, num_layers=2),
            # nn.LayerNorm(config.projector_dim)
        )
        
        # nn.Linear(original_dim, projector_dim)
       
        # 3. 融合模块（设为可训练）
        self.fusion_module = CrossAttentionFusion(query_dim=config.visual_feat_dim, context_dim=config.visual_feat_dim, num_heads=8, num_layers=2)
        self.asymmetric_fusion_module = AsymmetricFusion(dim=config.visual_feat_dim)

        # 4. 视频 caption 模型，用于接收 soft prompt（融合后的 embedding）计算 loss 或生成 caption
        
        # 视觉特征到LLM embedding空间的投影层 (可训练)
        self.llm_embed_dim = config.hidden_size
        # soft prompt长度 (query数量)
        # self.vis_proj = nn.Sequential(
        #                            nn.Linear(config.visual_feat_dim, self.llm_embed_dim),
        #                            nn.LayerNorm(self.llm_embed_dim))
        
        # self.vis_proj = nn.Sequential(
        #                            nn.Linear(config.visual_feat_dim, self.llm_embed_dim),
        #                            )
        self.soft_prompt_len = config.soft_prompt_len
        
        # self.vis_proj = VisualSoftPromptEncoder(
        #     input_dim=config.visual_feat_dim,     # e.g. 512
        #     output_dim=self.llm_embed_dim,        # e.g. 768
        #     prompt_len=self.soft_prompt_len,   # 你要生成的 soft prompt 长度
        #     num_layers=2,                         # 可调
        #     num_heads=8
        # )

        self.vis_proj = QFormerPromptEncoder(
            visual_dim=config.visual_feat_dim,     # e.g. 512
            prompt_dim=self.llm_embed_dim,        # e.g. 768
            num_query=self.soft_prompt_len,   # 你要生成的 soft prompt 长度
            
        )

    def forward(self, frames_images, video_embeddings):
        """
        参数说明：
          frames_images: Tensor，形状 [B, num_frames, 3, H, W]（视频帧预先处理成批次输入）
          videos: list，长度为 B，每个元素为视频路径字符串（供 video_backbone 提取 embedding）
          captions: 可选 list，长度为 B，用于训练时计算 loss
        返回：
          如果 captions 不为 None，则返回 caption 模型计算的 loss；
          否则返回生成的 caption 文本列表。
        """
        B, num_frames = frames_images.shape[:2]

        # 1. 计算视频 embedding
        
        video_embedding = self.projector(video_embeddings.squeeze(1))

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
        video_feats = [nf[-1] for nf in neck_feats]
        image_feats_neck = [nf[:-1] for nf in neck_feats]

        # 4. 进一步从 bbox_head 中获得多尺度的图像特征表示
        with torch.no_grad():
            # image_feats_processed 为列表，每个样本为一个列表（长度 num_frames），每帧包含多尺度特征（列表长度 num_scales）
            image_feats_processed = [self.detector.bbox_head.get_frame_emd(feat) for feat in image_feats_neck]

        # 假定所有样本、所有帧的尺度数量一致，取第一个样本第一个帧的尺度数作为 num_scales
        num_scales = len(image_feats_processed[0][0])
        fused_embeddings_list = []

        # 5. 对每个样本聚合多尺度特征、融合视频和图像信息
        for i in range(B):
            scale_features = []
            for scale in range(num_scales):
                # 对于样本 i，每一帧该尺度特征做全局平均池化，得到形状 [C]
                frame_scale_feats = []
                for j in range(num_frames):
                    feat = image_feats_processed[i][j][scale]
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0))
                    frame_scale_feats.append(pooled)
                # 合并为 [num_frames, C]
                frame_scale_feats = torch.stack(frame_scale_feats, dim=0)
                scale_features.append(frame_scale_feats)
            # 对所有尺度特征取均值，得到该样本每帧的聚合特征，形状 [num_frames, C]
            # frame_feats_aggregated = torch.stack(scale_features, dim=0).mean(dim=0)
            # 对该样本 video_feats 进行帧均值，得到视频级别特征，shape [1, C]
            video_feat_i = video_feats[i].mean(dim=0, keepdim=True)
            # 使用 AsymmetricFusion 对图像特征进行引导融合
            # fused_frame = self.asymmetric_fusion_module(video_feat_i, frame_feats_aggregated)
            # 使用 Cross-AttentionFusion 将视频特征与图像融合特征进一步融合，输出 shape 例如 [1, num_frames, visual_feat_dim]
            fused_embedding = self.fusion_module(video_feat_i, video_feat_i)
            fused_embeddings_list.append(fused_embedding)
        
        # 合并所有样本，形状 [B, soft_prompt_len, visual_feat_dim] （这里 soft_prompt_len 对应 num_frames）
        fused_embeddings = torch.cat(fused_embeddings_list, dim=0)
        soft_prompt = self.vis_proj(fused_embeddings)
        # print('soft shape is ', soft_prompt.shape)
        del frames_images_flat, img_feats_raw
        torch.cuda.empty_cache()
        return fused_embeddings, soft_prompt


class PrefixEncoder(nn.Module):
    def __init__(self, prompt_len, hidden_size, num_layers, num_heads):
        super().__init__()
        self.prompt_len = prompt_len
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.head_dim   = hidden_size // num_heads
        # 映射 soft_prompt → 每层 K,V
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_layers * 2 * hidden_size)
        )

    def forward(self, soft_prompt):               # [B, Lp, D]
        B, Lp, D = soft_prompt.size()
        past = self.mlp(soft_prompt)              # [B, Lp, 2*L*D]
        past = past.view(B, Lp, self.num_layers*2, self.num_heads, self.head_dim)
        past = past.permute(2, 0, 3, 1, 4).split(2)   # tuple(len=L) of (k,v)
        return tuple((p[0], p[1]) for p in past)      # [(B,nh,Lp,hd), ...]

from transformers.models.opt import OPTForCausalLM

class MyOPTWithPrefix(OPTForCausalLM):
    def __init__(self, config, prefix_encoder):
        super().__init__(config)
        self.prefix_encoder = prefix_encoder

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is None:
            # 第一次生成步，注入 prefix 作为 KV
            soft_prompt = kwargs["soft_prompt"]  # [B, Lp, D]
            past_key_values = self.prefix_encoder(soft_prompt)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }


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
        visual_proj = visual_proj + self.visual_pos                     # [B, F, D]

        # 准备 query tokens
        query = self.query_tokens.expand(B, -1, -1)                     # [B, Nq, D]

        # Cross-attention：Query attend to visual features
        output = self.cross_attn(tgt=query, memory=visual_proj)        # [B, Nq, D]
        return output
        # return self.output_norm(output)


