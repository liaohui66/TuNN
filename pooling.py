# repr.pooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter # 导入 PyG 的 scatter 函数
from typing import List

from repr.mlp import MLP


class WeightedAttentionPooling(nn.Module):
    """
    加权注意力池化层。
    模拟原始代码中的 WeightedAttentionPooling。
    """
    def __init__(self, gate_nn_hidden_dims: List[int], # Gate MLP 隐藏层+输出层(1)维度
                       message_nn_hidden_dims: List[int], # Message MLP 隐藏层+输出层维度
                       input_dim: int, # 输入特征 x 的维度
                       activation: nn.Module = nn.SELU()): # MLP 使用的激活函数
        """
        Args:
            gate_nn_hidden_dims (List[int]): 门控 MLP 的隐藏层和最终输出(1)的维度列表。例: [32, 1]
            message_nn_hidden_dims (List[int]): 消息 MLP 的隐藏层和输出维度列表。例: [64, 128]
            input_dim (int): 输入特征 x 的维度。
            activation (nn.Module): MLP 中隐藏层使用的激活函数。
        """
        super().__init__()

        if not gate_nn_hidden_dims or gate_nn_hidden_dims[-1] != 1:
            raise ValueError("gate_nn_hidden_dims must end with dimension 1.")
        if not message_nn_hidden_dims:
            raise ValueError("message_nn_hidden_dims cannot be empty.")

        # 门控网络: 输入维度 = x 的维度, 输出维度 = 1, 最后一层不激活
        self.gate_nn = MLP(input_dim=input_dim,
                           hidden_dims=gate_nn_hidden_dims,
                           activation=activation,
                           activate_last=False) # Gate 输出通常不激活

        # 消息网络: 输入维度 = x 的维度, 输出维度 = message_nn_hidden_dims[-1]
        self.message_nn = MLP(input_dim=input_dim,
                              hidden_dims=message_nn_hidden_dims,
                              activation=activation,
                              activate_last=True) # 消息变换后通常需要激活 (原始 SimpleNetwork 在线性层后有激活)
                              # 如果原始 SimpleNetwork 最后一层线性输出不激活，这里改为 False

        # 可训练的权重指数 (模拟 self.pow)
        self.pow = nn.Parameter(torch.randn(1))

        self.eps = 1e-10 # 用于数值稳定性

    def forward(self, x: torch.Tensor, index: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征张量，形状 [N, input_dim]，N 是节点/边总数。
            index (torch.Tensor): 段索引张量，形状 [N]，指示每个 x 属于哪个图/段 (dtype=torch.long)。
            weights (torch.Tensor): 每个 x 对应的权重，形状 [N] 或 [N, 1]。

        Returns:
            torch.Tensor: 池化后的特征张量，形状 [num_segments, message_output_dim]。
        """
        # 确保 weights 是 [N, 1]
        weights = weights.view(-1, 1)
        if weights.shape[0] != x.shape[0]:
             raise ValueError(f"Shape mismatch: x has {x.shape[0]} elements but weights has {weights.shape[0]}.")
        if index.shape[0] != x.shape[0]:
             raise ValueError(f"Shape mismatch: x has {x.shape[0]} elements but index has {index.shape[0]}.")


        # 1. 计算门控分数
        gate = self.gate_nn(x) # [N, 1]

        # 2. 数值稳定的 Softmax (按段)
        #    计算每个段内的最大门控分数
        gate_max = scatter(src=gate, index=index, dim=0, reduce='max') # [num_segments, 1]
        # 广播减去每个段的最大值以提高稳定性
        gate = gate - gate_max[index] # [N, 1]

        # 应用权重和指数，然后取指数
        # 确保 pow 作用在非负权重上，如果权重可能为负需要处理
        # 原始代码是 weights ** self.pow，如果 weights 为负且 pow 非整数会出问题
        # 为安全起见，可以先将 weights clamp 到正数或加 epsilon
        safe_weights = torch.clamp(weights, min=self.eps)
        gate = (safe_weights ** self.pow) * torch.exp(gate) # [N, 1]

        # 计算分母 (每个段的门控分数总和)
        gate_sum = scatter(src=gate, index=index, dim=0, reduce='sum') # [num_segments, 1]

        # 归一化门控分数 (注意力权重)
        # 使用 index_select 或直接索引来广播分母
        gate = gate / (gate_sum[index] + self.eps) # [N, 1]

        # 3. 计算消息
        message = self.message_nn(x) # [N, message_output_dim]

        # 4. 加权聚合消息
        out = gate * message # 逐元素相乘 [N, message_output_dim]
        # 按段求和
        pooled_out = scatter(src=out, index=index, dim=0, reduce='sum') # [num_segments, message_output_dim]

        return pooled_out