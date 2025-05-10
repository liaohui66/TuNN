# repr.chemical_transform.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QKVAttentionTorchFunctional(nn.Module): # 临时占位符
    def __init__(self, dropout=0.0): super().__init__(); self.dropout = dropout
    def forward(self, q, k, v): return v

class ChemicalTransformTorch(nn.Module):
    """
    PyTorch 版本的 chemical_transform 函数 (修正版)。
    输入为节点/特征点级别表示 [N, input_dim]。
    """
    def __init__(self, input_dim: int, # final_vec 的特征维度
                       lstm_hidden_size: int = 256,
                       attention_dropout: float = 0.0, # QKVAttention 的 dropout
                       dropout_rate: float = 0.2):    # LSTM 输入前的 dropout
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_rate = dropout_rate

        # 1. 自注意力模块 (使用函数式版本)
        self.self_attention = QKVAttentionTorchFunctional(dropout=attention_dropout)
        attention_output_dim = input_dim # 自注意力不改变维度

        # 2. Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 3. LSTM 层
        # --- [修正] 确保 input_size 正确 ---
        lstm_input_dim = attention_output_dim + input_dim # = 2 * input_dim
        # --- 修正结束 ---
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=False)
        
    def forward(self, final_vec: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            final_vec (torch.Tensor): 输入特征向量，形状 [N, input_dim]，N 是批次中总节点数。

        Returns:
            torch.Tensor: LSTM 的输出序列，形状 [N, lstm_hidden_size]。
        """
        if final_vec.numel() == 0:
            return torch.zeros((0, self.lstm_hidden_size), device=final_vec.device, dtype=final_vec.dtype)

        N = final_vec.shape[0]

        # --- 模拟 _concat_nbrs ---
        # 1.1 自注意力计算
        # 将输入 reshape 为 [batch=1, seq_len=N, input_dim]
        final_vec_for_attn = final_vec.unsqueeze(0)
        # 注意：QKVAttentionTorchFunctional 不处理 masking (presence)，如果需要 mask 需要在这里添加
        attn_output = self.self_attention(final_vec_for_attn, final_vec_for_attn, final_vec_for_attn)
        attn_output = attn_output.squeeze(0) # 变回 [N, input_dim]

        # 1.2 拼接
        total_fea = torch.cat([attn_output, final_vec], dim=-1) # [N, 2 * input_dim]
        # --- _concat_nbrs 结束 ---

        # 2. Dropout
        nbr_vec = self.dropout(total_fea) # [N, 2 * input_dim]

        # 3. 准备 LSTM 输入 [seq_len=N, batch=1, input_size]
        lstm_input = nbr_vec.unsqueeze(1)

        # 4. 初始化 LSTM 状态 [num_layers*dir, batch=1, hidden_size]
        h_0 = torch.zeros(1, 1, self.lstm_hidden_size, device=final_vec.device, dtype=final_vec.dtype)
        c_0 = torch.zeros(1, 1, self.lstm_hidden_size, device=final_vec.device, dtype=final_vec.dtype)
        initial_state = (h_0, c_0)

        # 5. 应用 LSTM
        lstm_output_seq, (h_n, c_n) = self.lstm(lstm_input, initial_state)

        # 6. 返回 LSTM 输出序列，移除假的 batch 维度 [N, lstm_hidden_size]
        return lstm_output_seq.squeeze(1)