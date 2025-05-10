# repr.descriptor.py
import torch
import torch.nn as nn
from typing import List

from repr.mlp import MLP
from repr.message import MessageLayerTorch
from repr.pooling import WeightedAttentionPooling

class DescriptorNetworkTorch(nn.Module):
    """PyTorch 版 DescriptorNetwork"""
    def __init__(self, elem_emb_len: int, # 嵌入查找后的维度 (例如 128)
                       elem_fea_len: int = 64, n_graph: int = 3,
                       elem_heads: int = 3, # MessageLayerTorch 当前未使用
                       elem_gate_hidden: List[int] = [256], # MessageLayer 内部 GlobalAttn gate MLP 隐藏维度
                       elem_msg_hidden: List[int] = [256],  # MessageLayer 内部 GlobalAttn msg MLP 隐藏维度
                       cry_heads: int = 3,
                       cry_gate_hidden: List[int] = [256], # 最终 cry_pool gate MLP 隐藏维度
                       cry_msg_hidden: List[int] = [256],  # 最终 cry_pool msg MLP 隐藏维度
                       activation: nn.Module = nn.SELU()):
        super().__init__()
        self.elem_fea_len = elem_fea_len
        self.n_graph = n_graph
        self.cry_heads = cry_heads

        # 1. 初始线性层 + 拼接权重 (在 forward 中处理)
        self.initial_linear = nn.Linear(elem_emb_len, elem_fea_len - 1, bias=True)
        self.activation = activation

        # 2. 消息传递层 (MessageLayerTorch 列表)
        # 确认 MessageLayerTorch 内部 MLP 的维度配置
        # gate_nn 输出是 1
        gate_dims_msg_layer = elem_gate_hidden + [1]
        # message_nn 输出必须是 elem_fea_len
        msg_dims_msg_layer = elem_msg_hidden + [elem_fea_len]

        self.message_layers = nn.ModuleList([
            MessageLayerTorch(
                elem_fea_len=elem_fea_len,
                elem_heads=elem_heads,
                elem_gate_hidden=gate_dims_msg_layer,
                elem_msg_hidden=msg_dims_msg_layer,
                activation=activation
            )
            for _ in range(n_graph)
        ])


        # 3. 多头晶体池化层 (WeightedAttentionPooling 列表)
        self.cry_pooling_layers = nn.ModuleList()
        for _ in range(cry_heads):
            # 确认最终池化层的 MLP 配置
            gate_dims_cry_pool = cry_gate_hidden + [1]
            msg_dims_cry_pool = cry_msg_hidden + [elem_fea_len]

            self.cry_pooling_layers.append(
                WeightedAttentionPooling(gate_nn_hidden_dims=gate_dims_cry_pool,
                                         message_nn_hidden_dims=msg_dims_cry_pool,
                                         input_dim=elem_fea_len, # 输入是消息传递后的 elem_fea
                                         activation=activation)
            )

    def forward(self, elem_weights: torch.Tensor, # [N, 1] or [N], N=总节点数
                      elem_fea_in: torch.Tensor,   # [N, elem_emb_len=128], 来自 embedding_lookup
                      batch: torch.Tensor          # [N], 节点到图的映射
                      ) -> torch.Tensor:         # 输出图级别特征 [batch_size, elem_fea_len]

        # 确保 weights 是 [N, 1]
        elem_weights = elem_weights.view(-1, 1)

        # 1. 初始线性变换
        elem_fea = self.initial_linear(elem_fea_in) # [N, elem_fea_len - 1]
        # 原始代码在这里没有激活

        # 2. 拼接权重
        elem_fea = torch.cat([elem_fea, elem_weights], dim=-1) # [N, elem_fea_len]

        # 3. 消息传递循环
        for i in range(self.n_graph):
            # MessageLayerTorch 接收权重(未使用)、特征和 batch 索引
            elem_fea = self.message_layers[i](elem_weights, elem_fea, batch)

        # 4. 多头图池化
        head_outputs = []
        for i in range(self.cry_heads):
            # WeightedAttentionPooling 接收节点特征、批次索引和节点权重
            pooled_fea = self.cry_pooling_layers[i](x=elem_fea, index=batch, weights=elem_weights)
            head_outputs.append(pooled_fea)

        # 5. 平均头输出
        final_cry_fea = torch.mean(torch.stack(head_outputs, dim=0), dim=0) # [batch_size, elem_fea_len]

        return final_cry_fea