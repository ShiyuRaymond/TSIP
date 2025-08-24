import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class CrossAttentionFusion(BaseModule):
    def __init__(self, 
                 query_dim=512,       # video embedding维度
                 context_dim=512,     # YOLO-WORLD的image feature维度
                 num_heads=8,         # 注意力头数
                 num_layers=4,        # Transformer层数
                 dropout=0.1,
                 num_prompt=32):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        
        # self.query_tokens = nn.Parameter(torch.randn(1, num_prompt, query_dim))


        # Transformer decoder层实现cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(512))
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, )
        

        # 投影context到query的维度（若不一致时使用）
        if context_dim != query_dim:
            self.context_proj = nn.Linear(context_dim, query_dim)
        else:
            self.context_proj = nn.Identity()

    def forward(self, video_queries, frame_features, frame_padding_mask=None):
        """
        参数：
            video_queries: [B, num_queries, query_dim] - 来自VideoCLIP
            frame_features: [B, num_frames, context_dim] - YOLO的每帧特征
            frame_padding_mask: [B, num_frames] - 用于mask掉padding frame（可选）
        返回：
            fused_embedding: [B, num_queries, query_dim]
        """
        # 如果需要，project frame_features维度到query_dim
        frame_features = self.context_proj(frame_features)  # [B,num_frames,query_dim]
        B,_,_ = video_queries.shape
                           # [B, Nq, D]

        # Transformer decoder forward
        fused_embedding = self.transformer_decoder(
            tgt=video_queries ,                   # 作为Query
            memory=frame_features,               # 作为Key和Value
            memory_key_padding_mask=frame_padding_mask
        )

        return fused_embedding  # [B, num_queries, query_dim]
    
    
    
@MODELS.register_module()    
class AsymmetricFusion(BaseModule):
    # def __init__(self, dim):
    #     super().__init__()
    #     self.gate_layer = nn.Linear(dim, dim)

    # def forward(self, temporal_feat, spatial_feat):
    #     """
    #     Args:
    #         temporal_feat: [B, num_queries, dim] 视频级特征 (主导特征)
    #         spatial_feat: [B, num_frames, dim] 帧级空间特征 (被引导特征)
    #     Returns:
    #         fused_feat: [B, num_frames, dim] 融合后的特征
    #     """
    #     # 首先对temporal特征进行全局池化，得到[B, 1, dim]引导特征
    #     temporal_global = temporal_feat.mean(dim=1, keepdim=True)  # [B, 1, dim]

    #     # 计算门控
    #     gate = torch.sigmoid(self.gate_layer(temporal_global))  # [B, 1, dim]

    #     # 执行非对称门控融合
    #     fused_feat = spatial_feat  # [B, num_frames, dim]

    #     return fused_feat
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        """
        query:   [B, Tq, D] - e.g., temporal features
        context: [B, Tc, D] - e.g., global or fused features
        return:  [B, Tq, D] - fused result
        """
        attn_output, _ = self.cross_attn(query=query, key=context, value=context)
        output =self.norm(query + self.dropout(attn_output))
        return output




@MODELS.register_module()    
class ScaleAttention(BaseModule):
    def __init__(self, dim, num_heads=4, dropout=0.0, ff_ratio=4, num_layers=2):
        super().__init__()

        # 每层是一个 DecoderLayer（会用到 self-attn 和 ffn）
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ff_ratio,
            dropout=dropout,
            # activation='gelu',
            batch_first=True,
            # norm_first=False
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
            # norm=None  # 显式关闭最终 LayerNorm
        )

    def forward(self, x):
        # x: [num_frames, num_scales, C]  → 每帧是一个 batch，scale 是 token
        # 为了只用 self-attention，把 memory 设置为 x 自身
        # 这会触发 decoder 中的 self-attention（tgt）模块

        output = self.decoder(tgt=x, memory=x)  # self-attn only, acts like encoder
        output = output.mean(dim=1)  # 聚合尺度
        return output  # [num_frames, C]
