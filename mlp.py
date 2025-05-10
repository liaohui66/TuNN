# repr.mlp.py
import torch
import torch.nn as nn
from typing import List, Optional

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 activation: nn.Module = nn.SELU(),
                 activate_last: bool = False,
                 dropout_rate: Optional[float] = None):
        super().__init__()
        # [--- 使用 ModuleList 的版本 ---]
        self.layers = nn.ModuleList() # 使用 ModuleList
        current_dim = input_dim
        num_layers = len(hidden_dims)

        # print(f"\n--- Building MLP Layers (Input Dim: {input_dim}, Hidden Dims: {hidden_dims}) ---")
        for i, h_dim in enumerate(hidden_dims):
            if not isinstance(current_dim, int) or current_dim <= 0: raise ValueError(...)
            if not isinstance(h_dim, int) or h_dim <= 0: raise ValueError(...)

            linear_layer = nn.Linear(current_dim, h_dim)
            self.layers.append(linear_layer) # 添加 Linear
            # print(f"Layer {len(self.layers)-1}: {linear_layer}, Type: {type(linear_layer)}")

            is_last_layer = (i == num_layers - 1)
            if not is_last_layer or activate_last:
                if not isinstance(activation, nn.Module): raise TypeError(...)
                self.layers.append(activation) # 添加激活
                # print(f"Layer {len(self.layers)-1}: {activation}, Type: {type(activation)}")
                if dropout_rate is not None and dropout_rate > 0:
                    dropout_layer = nn.Dropout(p=dropout_rate)
                    self.layers.append(dropout_layer) # 添加 Dropout
                    # print(f"Layer {len(self.layers)-1}: {dropout_layer}, Type: {type(dropout_layer)}")

            current_dim = h_dim
        # print("--- Finished Building MLP Layers ---\n")

        # [--- BEGIN MODIFICATION ---]
        # --- 在 __init__ 末尾检查 self.layers 是否创建成功 ---
        if not hasattr(self, 'layers') or not isinstance(self.layers, nn.ModuleList) or len(self.layers) == 0:
             # print("ERROR: self.layers was not created or is empty in MLP.__init__!")
             # 可以抛出错误以停止执行
             raise RuntimeError("MLP layers initialization failed.")
        # else:
             # print(f"MLP.__init__ finished. self.layers created with {len(self.layers)} items.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [--- 添加调试打印 ---]
        # print(f"  >> Entering MLP forward for {self}, input shape: {x.shape}")
        # [--- 打印结束 ---]
        h = x
        if not hasattr(self, 'layers'):
             raise AttributeError("MLP object has no attribute 'layers'...")
        for i, layer in enumerate(self.layers):
            # [--- 添加调试打印 ---]
            # print(f"    MLP Layer {i}: {layer}")
            h_before = h
            # [--- 打印结束 ---]
            h = layer(h)
            # [--- 添加调试打印 ---]
            # print(f"      Input shape: {h_before.shape}, Output shape: {h.shape}")
            # [--- 打印结束 ---]
        # print(f"  << Exiting MLP forward for {self}, final output shape: {h.shape}")
        return h