# repr.cnn.py
import torch
import torch.nn as nn
from typing import List

class CNNEncoderTorch(nn.Module):
    """PyTorch version of the snt.nets.ConvNet2D used in make_capsule."""
    def __init__(self,
                 in_channels: int = 1, # 输入通道数 (来自 reshape)
                 output_channels: List[int] = [256, 256, 256, 256],
                 kernel_size: int = 3, # 假设所有层都是 3x3
                 strides: List[int] = [2, 2, 1, 1],
                 padding: str = 'same', # 对应 snt.SAME
                 activation: nn.Module = nn.SELU(),
                 activate_final: bool = True):
        print(f"Debug CNNEncoderTorch: output_channels={output_channels}, strides={strides}")  # 添加调试信息
        super().__init__()

        if not output_channels or not strides or len(output_channels) != len(strides):
            raise ValueError("output_channels and strides must be non-empty lists of the same length.")

        layers = []
        current_channels = in_channels
        num_layers = len(output_channels)

        for i in range(num_layers):
            # 计算 padding 值以模拟 'SAME'
            # 对于 stride=1, padding = (kernel_size - 1) // 2
            # 对于 stride=2, padding 可能需要更仔细计算以匹配 TF 行为，但 (k-1)//2 通常也接近
            # PyTorch Conv2d 的 padding 参数接受 int 或 tuple
            pad_val = (kernel_size - 1) // 2 if padding.lower() == 'same' else 0

            layers.append(nn.Conv2d(in_channels=current_channels,
                                    out_channels=output_channels[i],
                                    kernel_size=kernel_size,
                                    stride=strides[i],
                                    padding=pad_val,
                                    bias=True)) # Sonnet Conv2D 默认使用偏置

            # 添加激活函数 (除非是最后一层且 activate_final=False)
            is_last_layer = (i == num_layers - 1)
            if not is_last_layer or activate_final:
                layers.append(activation)

            current_channels = output_channels[i]

        self.network = nn.Sequential(*layers)

        self.out_channels = output_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状 [B, C_in, H, W] (例如 [N, 1, 16, 16])
        Returns:
            torch.Tensor: 输出特征图，形状 [B, C_out, H', W']
        """
        return self.network(x)
    
    def get_output_channels(self) -> int:
        return self.out_channels
