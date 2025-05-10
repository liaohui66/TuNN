# repr.stoi.py
import torch
import torch.nn as nn
from typing import List

from repr.descriptor import DescriptorNetworkTorch
from repr.mlp import MLP

class StoiRepTorch(nn.Module):
    """PyTorch 版 Stoi_Rep"""
    def __init__(self,
                 # Stoi_Rep 参数
                 n_target: int,
                 elem_emb_len: int, # 这个参数现在可能代表 Linear 层的输出维度 (128)
                 # DescriptorNetwork 参数
                 elem_fea_len: int = 64, n_graph: int = 3,
                 elem_heads: int = 3, elem_gate_hidden: List[int] = [256],
                 elem_msg_hidden: List[int] = [256], cry_heads: int = 3,
                 cry_gate_hidden: List[int] = [256], cry_msg_hidden: List[int] = [256],
                 # 输出 MLP 参数 (原始代码 Stoi_Rep 中没有显式输出 MLP)
                 # out_hidden: List[int] = [1024, 512, 256, 128, 64], # 原始参数，这里没用到
                 vocab_size: int = 103, # one-hot 输入维度
                 embedding_initializer_range: float = 0.05, # 线性层初始化范围
                 activation: nn.Module = nn.SELU()):
        super().__init__()
        self.n_target = n_target # 虽然原始代码没用，但保留接口

        # 1. 替代 embedding_lookup (Linear 层)
        self.embedding_linear = nn.Linear(vocab_size, elem_emb_len, bias=False)
        # 初始化权重
        with torch.no_grad():
            nn.init.trunc_normal_(self.embedding_linear.weight, std=embedding_initializer_range, a=-2*embedding_initializer_range, b=2*embedding_initializer_range)

        # 2. 实例化 DescriptorNetworkTorch
        self.material_nn = DescriptorNetworkTorch(
            elem_emb_len=elem_emb_len, # 嵌入后的维度
            elem_fea_len=elem_fea_len,
            n_graph=n_graph,
            elem_heads=elem_heads,
            elem_gate_hidden=elem_gate_hidden,
            elem_msg_hidden=elem_msg_hidden,
            cry_heads=cry_heads,
            cry_gate_hidden=cry_gate_hidden,
            cry_msg_hidden=cry_msg_hidden,
            activation=activation
        )

        # 3. (可选) 添加最终的输出 MLP
        # 原始 Stoi_Rep 直接返回 DescriptorNetwork 的输出
        # 如果需要映射到最终的 n_target 维度，可以在这里加
        # self.output_mlp = MLP(input_dim=elem_fea_len, hidden_dims=out_hidden + [n_target], ...)

    def forward(self, comp_w: torch.Tensor,    # [N, 1] or [N] 组成权重
                      atom_fea: torch.Tensor,  # [N, vocab_size=103] one-hot 编码
                      batch: torch.Tensor      # [N] 节点到图的映射 (替代 ele_idx)
                      # index1, index2 (fea, nbr) 不再需要
                      ) -> torch.Tensor:       # 输出图级别特征 [batch_size, elem_fea_len]

        # 1. 应用 Embedding (Linear 层)
        # 输入 atom_fea: [N, 103] -> 输出 ele_fea: [N, elem_emb_len=128]
        ele_fea = self.embedding_linear(atom_fea)

        # 2. 调用 DescriptorNetworkTorch
        # 输入: elem_weights [N, 1], ele_fea [N, 128], batch [N]
        # 输出: crys_fea [batch_size, elem_fea_len=64]
        crys_fea = self.material_nn(elem_weights=comp_w,
                                   elem_fea_in=ele_fea,
                                   batch=batch)

        # 3. (可选) 应用输出 MLP
        # if hasattr(self, 'output_mlp'):
        #     crys_fea = self.output_mlp(crys_fea)

        return crys_fea