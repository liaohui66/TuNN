# repr.batch_mlp_torch.py
import torch
import torch.nn as nn
from typing import List, Optional

from repr.mlp import MLP

class BatchMLPTorch(nn.Module):
    """
    PyTorch 版本的 BatchMLP (简化版)。
    使用标准的共享权重 MLP 替代原始的 BatchLinear。
    """
    def __init__(self, n_hiddens: List[int], # 隐藏层和输出层维度列表
                       activation: nn.Module = nn.SELU(),
                       activation_final: nn.Module = nn.SELU(), # 最后一层可选激活
                       activate_final: bool = False,
                       use_bias: bool = True, # 控制最后一层是否使用偏置
                       input_dim: Optional[int] = None, # 可选：提前指定输入维度
                       **kwargs): # 接收并忽略其他参数
        super().__init__()
        if not n_hiddens: raise ValueError("n_hiddens cannot be empty")
        self._n_hiddens = n_hiddens if isinstance(n_hiddens, list) else [n_hiddens]
        self._activation = activation
        self._activate_final = activate_final
        self._use_bias_final = use_bias # 只控制最后一层

        self.network: Optional[nn.Module] = None # 可以是 MLP 或 Sequential
        self._input_dim = input_dim
        self._initialized = False

        if self._input_dim is not None:
            self._build_network(self._input_dim)

    def _build_network(self, input_dim: int):
        """构建内部网络"""
        if MLP is not None: # 优先使用我们定义的 MLP 类
             self.network = MLP(input_dim=input_dim,
                                hidden_dims=self._n_hiddens,
                                activation=self._activation,
                                activate_last=self._activate_final)
             # 注意: 当前 MLP 类没有精细控制最后一层偏置的选项
             if not self._use_bias_final:
                  # --- 访问 MLP 实例的 self.layers 属性 ---
                  if hasattr(self.network, 'layers') and isinstance(self.network.layers, nn.ModuleList) and len(self.network.layers) > 0:
                      last_layer_or_activation = self.network.layers[-1]
                      actual_last_linear_layer = None

                      if isinstance(last_layer_or_activation, nn.Linear):
                          # 如果最后一项就是 Linear 层
                          actual_last_linear_layer = last_layer_or_activation
                      elif len(self.network.layers) > 1 and isinstance(self.network.layers[-2], nn.Linear):
                          # 如果最后一项是激活/Dropout，检查倒数第二项是否是 Linear
                          actual_last_linear_layer = self.network.layers[-2]

                      if actual_last_linear_layer is not None:
                          # print(f"BatchMLPTorch: Setting final linear layer bias to False for output dim {self._n_hiddens[-1]}.")
                          # 直接设置 bias 为 None (如果参数存在)
                          if hasattr(actual_last_linear_layer, 'bias') and actual_last_linear_layer.bias is not None:
                              actual_last_linear_layer.bias = None
                              # 注意：直接设置 bias 为 None 可能在某些 PyTorch 版本或特定优化器下有问题
                              # 更安全的方式可能是重新初始化 Linear 层时不带 bias
                              # 但对于临时修复，设置为 None 通常可行
                          else:
                              print("  (Bias already None or not applicable)")
                      else:
                          print("BatchMLPTorch: Could not find final Linear layer in MLP to set bias=False.")
                  else:
                      print("BatchMLPTorch: Could not access 'layers' in MLP instance to set bias=False.")


        else: # 如果 MLP 类不可用，退回到使用 nn.Sequential
             print("BatchMLPTorch: MLP class not found, using nn.Sequential.")
             layers = []
             current_dim = input_dim
             num_layers = len(self._n_hiddens)
             for i, h_dim in enumerate(self._n_hiddens):
                 is_last_layer = (i == num_layers - 1)
                 use_bias_current = True if not is_last_layer else self._use_bias_final
                 layers.append(nn.Linear(current_dim, h_dim, bias=use_bias_current))
                 if not is_last_layer or self._activate_final:
                      # 原始的 BatchMLP 最后才判断是否加 final activation
                      # 这里的逻辑是在每层（除了最后一层，除非 activate_final）后加 activation
                      if is_last_layer:
                          layers.append(self._activation_final)
                      else:
                          layers.append(self._activation)
                 current_dim = h_dim
             self.network = nn.Sequential(*layers)

        self._initialized = True
        # print(f"BatchMLPTorch initialized with input dim {input_dim}, output dim {self._n_hiddens[-1]}.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
             if x.dim() < 2: raise ValueError(f"Input tensor x must have >= 2 dims, got {x.shape}")
             input_dim = x.shape[-1]
             self._build_network(input_dim)
             if self.network is not None:
                 self.network = self.network.to(x.device)

        if self.network is None: raise RuntimeError("Network not initialized in BatchMLPTorch.")
        return self.network(x)