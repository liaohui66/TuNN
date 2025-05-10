# repr.set_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import List, Tuple, Optional

class SetTransformerTorch(nn.Module):
    """
    PyTorch 版本的 set_transformer (基于 LSTM 和注意力聚合)。
    增强了数值稳定性和梯度检查。
    """
    def __init__(self, input_dim: int, n_hidden: int = 512, loop: int = 3,
                 lstm_activation=torch.tanh):
        super().__init__()
        self.n_hidden = n_hidden
        self.loop = loop
        self.lstm_activation = lstm_activation
        self.input_to_m = nn.Linear(input_dim, n_hidden)
        # LSTMCell 输入是 q_star(2*n_hidden), 输出是 h(n_hidden), c(n_hidden)
        self.lstm_cell = nn.LSTMCell(input_size=2 * n_hidden, hidden_size=n_hidden)
        # 增大 epsilon 提高稳定性
        self.eps = 1e-7

    def _check_nan_inf(self, tensor: torch.Tensor, name: str, loop_idx: int, batch_idx_for_debug: Optional[int] = None) -> bool:
        """辅助函数：检查 NaN 或 Inf 并打印警告"""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        if has_nan or has_inf:
            batch_info = f" (Batch {batch_idx_for_debug})" if batch_idx_for_debug is not None else ""
            status = []
            if has_nan: status.append("NaN")
            if has_inf: status.append("Inf")
            print(f"CRITICAL WARNING: {'/'.join(status)} detected in '{name}' (loop {loop_idx}){batch_info}!")
            # 可以在这里添加更多调试信息，例如打印出问题的 tensor 值
            # print(tensor)
            return True
        return False
    
    def forward(self, features: torch.Tensor, feature_graph_index: torch.Tensor, batch_idx_for_debug: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): 输入特征 [N, input_dim]。
            feature_graph_index (torch.Tensor): 图索引 [N]。
            batch_idx_for_debug (Optional[int]): (可选) 用于打印警告的批次索引。
        """
        feature_graph_index = feature_graph_index.long()
        device = features.device
        dtype = features.dtype

        num_graphs = 0
        if feature_graph_index.numel() > 0:
            num_graphs = int(feature_graph_index.max().item() + 1)
        else:
            return torch.zeros((0, 2 * self.n_hidden), device=device, dtype=dtype)


        # 1. 初始变换
        # 确保输入 features 需要梯度
        if not features.requires_grad:
             # print(f"CRITICAL WARNING (SetTF Start): Input 'features' does not require grad! Grad chain likely broken before this module (Batch {batch_idx_for_debug}).")
             # 如果输入就没梯度，后续也无法产生梯度
             features = features.detach() # 明确分离，避免后续出错

        m = self.input_to_m(features) # [N, n_hidden]
        if self._check_nan_inf(m, "m (initial)", -1, batch_idx_for_debug):
             return torch.zeros((num_graphs, 2 * self.n_hidden), device=device, dtype=dtype).requires_grad_(features.requires_grad) # 返回与输入匹配的 requires_grad

        # 2. 初始化 LSTM 状态和 q_star
        h = torch.zeros(num_graphs, self.n_hidden, device=device, dtype=dtype)
        c = torch.zeros(num_graphs, self.n_hidden, device=device, dtype=dtype)
        q_star = torch.zeros(num_graphs, 2 * self.n_hidden, device=device, dtype=dtype)
        # 初始状态不需要梯度

        # 3. 迭代更新循环
        for i in range(self.loop):
            h_prev, c_prev = h, c
            q_star_prev = q_star

            if self._check_nan_inf(q_star_prev, f"q_star_prev", i, batch_idx_for_debug): break
            if self._check_nan_inf(h_prev, f"h_prev", i, batch_idx_for_debug): break
            if self._check_nan_inf(c_prev, f"c_prev", i, batch_idx_for_debug): break

            if h_prev.numel() > 0 and c_prev.numel() > 0:
                 h, c = self.lstm_cell(q_star_prev, (h_prev, c_prev))
                 if self._check_nan_inf(h, f"h (LSTM output)", i, batch_idx_for_debug): break
                 if self._check_nan_inf(c, f"c (LSTM output)", i, batch_idx_for_debug): break
            else:
                 h = torch.zeros_like(h_prev); c = torch.zeros_like(c_prev)

            # 检查 LSTM 输出是否意外丢失梯度
            if not h.requires_grad and q_star_prev.requires_grad and num_graphs > 0:
                 print(f"WARNING (SetTF Loop {i}): LSTM output 'h' requires_grad=False despite input q_star requiring grad!")

            # 注意力计算
            h_broadcast = h[feature_graph_index]
            e_i_t = torch.sum(m * h_broadcast, dim=-1, keepdim=True)
            if self._check_nan_inf(e_i_t, f"e_i_t", i, batch_idx_for_debug): break

            # 按图进行 Softmax (增强稳定性)
            with torch.no_grad():
                e_i_t_max = scatter(src=e_i_t.detach(), index=feature_graph_index, dim=0, reduce='max', dim_size=num_graphs)
                e_i_t_max = torch.nan_to_num(e_i_t_max, nan=-1e10, posinf=-1e10, neginf=-1e10) # 替换 Inf

            e_i_t = e_i_t - e_i_t_max[feature_graph_index]
            if self._check_nan_inf(e_i_t, f"e_i_t after sub max", i, batch_idx_for_debug): break

            # 使用 clamp 限制 exp 输入
            exp_e = torch.exp(torch.clamp(e_i_t, min=-30, max=20)) # 限制下限防止 exp(过小负数) 为 0
            if self._check_nan_inf(exp_e, f"exp_e", i, batch_idx_for_debug): break

            exp_sum = scatter(src=exp_e, index=feature_graph_index, dim=0, reduce='sum', dim_size=num_graphs)
            if self._check_nan_inf(exp_sum, f"exp_sum", i, batch_idx_for_debug): break

            # 检查分母
            denominator = exp_sum[feature_graph_index] + self.eps
            if (denominator < self.eps / 2).any():
                 print(f"WARNING: Near-zero denominator detected in attention (loop {i}, Batch {batch_idx_for_debug})!")
                 # 可以考虑在这里添加处理，例如给 a_i_t 一个默认值，但可能会影响学习
            a_i_t = exp_e / denominator
            if self._check_nan_inf(a_i_t, f"a_i_t", i, batch_idx_for_debug): break

            # 加权聚合
            weighted_m = m * a_i_t
            r_t = scatter(src=weighted_m, index=feature_graph_index, dim=0, reduce='sum', dim_size=num_graphs)
            if self._check_nan_inf(r_t, f"r_t", i, batch_idx_for_debug): break

            # 更新 q_star
            q_star = torch.cat([h, r_t], dim=-1)
            if self._check_nan_inf(q_star, f"q_star", i, batch_idx_for_debug): break

            # 在循环结束时检查 q_star 的梯度状态
            if not q_star.requires_grad and m.requires_grad and num_graphs > 0: # 只有当 m 有梯度时，丢失梯度才是个问题
                 print(f"ERROR (SetTF Loop {i}, Batch {batch_idx_for_debug}): q_star lost gradient connection!")
                 # break # 如果丢失梯度就提前退出

        # 最终检查
        if self._check_nan_inf(q_star, "FINAL q_star", -1, batch_idx_for_debug):
             q_star = torch.zeros_like(q_star) # 如果最终结果是 NaN/Inf，返回零

        if not q_star.requires_grad and m.requires_grad and num_graphs > 0: # 再次检查最终结果
            print(f"ERROR (SetTF End, Batch {batch_idx_for_debug}): FINAL q_star does not require grad!")

        # print(f"  DEBUG (SetTF End): FINAL q_star requires_grad={q_star.requires_grad}") # 可以保留这个用于确认
        return q_star