# repr.message.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.glob import GlobalAttention # 导入 PyG 的 GlobalAttention
from typing import List

from repr.mlp import MLP

class MessageLayerTorch(nn.Module):
    """
    PyTorch 版本的 MessageLayer，使用 GlobalAttention 替代原始的 gather+pooling。
    注意：此实现忽略了原始代码中基于 self_fea_idx/nbr_fea_idx 的复杂 gather 逻辑，
         并假设 GlobalAttention 可以学习到相似的全局上下文表示。
         它也暂时忽略了 elem_weights 对消息传递的直接影响。
    """
    def __init__(self, elem_fea_len: int,
                       elem_gate_hidden: List[int],    # GlobalAttention gate_nn 隐藏层维度
                       elem_msg_hidden: List[int],     # GlobalAttention message_nn 隐藏层维度
                       activation: nn.Module = nn.SELU(),
                       **kwargs): # 接收并忽略 elem_heads 等额外参数
        """
        Args:
            elem_fea_len (int): 输入和输出的元素特征维度。
            elem_gate_hidden (List[int]): GlobalAttention 门控 MLP 的隐藏维度列表。
            elem_msg_hidden (List[int]): GlobalAttention 消息 MLP 的隐藏维度列表。
            activation (nn.Module): MLP 使用的激活函数。
        """
        super().__init__()
        self.elem_fea_len = elem_fea_len

        # --- 使用 PyG 的 GlobalAttention ---
        # Gate MLP 输出维度为 1
        gate_dims = elem_gate_hidden + [1]
        gate_mlp = MLP(input_dim=elem_fea_len,
                       hidden_dims=gate_dims,
                       activation=activation,
                       activate_last=False) # 注意力 gate 通常不激活

        # Message MLP 输出维度必须是 elem_fea_len (为了残差连接)
        msg_dims = elem_msg_hidden + [elem_fea_len]
        message_mlp = MLP(input_dim=elem_fea_len,
                          hidden_dims=msg_dims,
                          activation=activation,
                          activate_last=True) # 消息变换后激活

        self.global_attention_pooling = GlobalAttention(gate_nn=gate_mlp, nn=message_mlp)

        # 可选：添加 LayerNorm
        self.layer_norm = nn.LayerNorm(elem_fea_len)


    def forward(self, elem_weights: torch.Tensor, # [N, 1] or [N], N=总节点数 (当前未使用)
                      elem_in_fea: torch.Tensor,   # [N, elem_fea_len]
                      batch: torch.Tensor         # [N], 节点到图的映射 (替代 self_fea_idx?)
                      # nbr_fea_idx 不再需要
                      ) -> torch.Tensor:        # 输出更新后的节点特征 [N, elem_fea_len]
        """
        Args:
            elem_weights (torch.Tensor): (未使用) 元素权重。
            elem_in_fea (torch.Tensor): 输入元素特征。
            batch (torch.Tensor): 节点到图的映射索引。

        Returns:
            torch.Tensor: 更新后的元素特征。
        """

        # 1. 计算全局上下文向量
        #    GlobalAttention(x, batch) 返回 [batch_size, nn_output_dim]
        global_context = self.global_attention_pooling(elem_in_fea, batch)

        # 2. 将全局上下文广播回每个节点
        #    global_context[batch] -> [N, nn_output_dim] (nn_output_dim 应为 elem_fea_len)
        if global_context.shape[-1] != elem_in_fea.shape[-1]:
             # 如果维度不匹配（例如 message_mlp 配置错误），则无法直接相加
             # 应该在 __init__ 中确保维度一致，这里可以加个断言或错误
             raise ValueError(f"Dimension mismatch in MessageLayerTorch: "
                              f"Global context ({global_context.shape[-1]}) != "
                              f"Input features ({elem_in_fea.shape[-1]})")
        nodes_global_context = global_context[batch]

        # 3. 残差连接
        #    将全局信息加到原始特征上
        updated_fea = elem_in_fea + nodes_global_context

        # 4. 可选：应用 LayerNorm
        updated_fea = self.layer_norm(updated_fea)

        # 5. 可以考虑再加一个激活函数？原始代码似乎没有
        # updated_fea = F.selu(updated_fea)

        return updated_fea