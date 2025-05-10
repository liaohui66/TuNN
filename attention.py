# repr.attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QKVAttentionTorchFunctional(nn.Module):
    """使用 F.scaled_dot_product_attention 模拟 QKVAttention"""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = dropout

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                presence_mask: torch.Tensor = None) -> torch.Tensor:
        attn_mask = None
        if presence_mask is not None:
             if presence_mask.dim() == 2: # [B, N] -> [B, 1, N]
                 attn_mask = presence_mask.unsqueeze(1)
             elif presence_mask.dim() == 3: # [B, M, N]
                 attn_mask = presence_mask
             else:
                  print("Warning: Invalid presence_mask dimension. Ignoring.")
             #确保 mask 是布尔类型，True 表示保留
             if attn_mask is not None and attn_mask.dtype != torch.bool:
                 attn_mask = attn_mask.bool()


        # F.scaled_dot_product_attention 在 PyTorch 较新版本中可以直接处理布尔 mask
        # (True = 保留, False = mask掉)
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask, # None or Boolean mask
            dropout_p=self.dropout_p if self.training else 0.0
        )
        return attn_output