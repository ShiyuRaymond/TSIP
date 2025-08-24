import torch
import torch.nn as nn
from mmyolo.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class VisualSoftPromptEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,           # fused visual feature dim
                 output_dim: int,          # LLM embedding dim
                 prompt_len: int = 10,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 ff_dim: int = 2048):
        super().__init__()
        self.prompt_len = prompt_len
        self.output_dim = output_dim

        # 映射到 LLM embedding 空间
        self.input_proj = nn.Linear(input_dim, output_dim)

        # MLP 类似 PrefixEncoder 的初始映射增强
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)
        )

        # learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, prompt_len, output_dim))

        # Transformer 编码器增强 soft prompt 表达
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最后 LayerNorm（可选）
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feature: [B, prompt_len, input_dim]
        Returns:
            soft_prompt: [B, prompt_len, output_dim]
        """
        x = self.input_proj(fused_feature)           # 映射到 output_dim
        x = self.mlp(x)                              # 类似 prefix encoder 的 MLP 增强
        x = x + self.pos_embed[:, :x.size(1)]        # 加位置编码
        x = self.encoder(x)                          # self-attention 强化
        x = self.norm(x)                             # 最终 LayerNorm 稳定性更强
        return x

    
@MODELS.register_module()
class PrefixEncoder(nn.Module):
    def __init__(self, prompt_len, hidden_size, num_layers=2, num_heads=8):
        super().__init__()
        self.prompt_len = prompt_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # learnable position embedding
        # self.pos_embed = nn.Parameter(torch.randn(1, 32, 2048))

        # 原本的 MLP，用来先映射特征
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 自注意力层，提升 soft prompt 自身表达
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True  # 和 BLIP-2 一致
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP → 生成 past_key_values
        self.out_proj = nn.Linear(hidden_size, num_layers * 2 * hidden_size)

        # 额外的 Norm 层
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, soft_prompt):
        """
        Args:
            soft_prompt: [B, Lp, D]
        Returns:
            List of past_key_values: len = num_layers, each (k,v) shape: [B, nh, Lp, hd]
        """
        B, Lp, D = soft_prompt.size()

        x = self.mlp(soft_prompt)  # [B, Lp, D]
        # x = x + self喷水.pos_embed[:, :Lp, :]          # 添加位置编码
        x = self.transformer_encoder(x)  # self-attention增强
        x = self.norm(x)  # LayerNorm 稳定训练

        past = self.out_proj(x)  # [B, Lp, 2*L*D]
        past = past.view(B, Lp, self.num_layers * 2, self.num_heads, self.head_dim)
        past = past.permute(2, 0, 3, 1, 4).split(2)  # tuple of (k,v) per layer
        return tuple((p[0], p[1]) for p in past)
    
    
from transformers.models.opt import OPTForCausalLM

@MODELS.register_module()
class PrefixOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config, prefix_encoder: PrefixEncoder):
        super().__init__(config)
        self.prefix_encoder = prefix_encoder

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is None:
            if "soft_prompt" not in kwargs:
                raise ValueError("Missing soft_prompt for prefix tuning")
            soft_prompt = kwargs.pop("soft_prompt")  # ✅ pop 掉
            past_key_values = self.prefix_encoder(soft_prompt)

        # ✅ 过滤掉 cache_position
        kwargs.pop("cache_position", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            **kwargs
        }


    def forward(self, input_ids=None, attention_mask=None, labels=None,
                soft_prompt=None, past_key_values=None, **kwargs):
        if past_key_values is None and soft_prompt is not None:
            past_key_values = self.prefix_encoder(soft_prompt)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs
        )