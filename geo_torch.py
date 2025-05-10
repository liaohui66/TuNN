# geometry.geo_torch.py
import torch
import torch.nn.functional as F
import math
from typing import Optional

def geometric_transform_torch(pose_tensor: torch.Tensor,
                              n_caps: int, # 现在确认这个是必须的
                              n_votes: int, # 这个也是必须的
                              similarity: bool = False,
                              nonlinear: bool = True,
                              as_matrix: bool = False) -> torch.Tensor:
    """
    PyTorch version of geometric_transform. Converts 6D pose vector to 3x3 matrix or returns 6D vector.

    Args:
        pose_tensor (Tensor): Input tensor with shape [..., 6].
        n_caps (Optional[int]): Number of capsules. Required if as_matrix=True
                                to correctly reshape. Corresponds to the
                                dimension before n_votes in the original code's logic.
        n_votes (Optional[int]): Number of votes. Required if as_matrix=True
                                 to correctly reshape. Corresponds to the
                                 dimension before the params (6) in the original code's logic.
        similarity (bool): If True, compute similarity transform. Defaults to False.
        nonlinear (bool): If True, apply nonlinearities (SELU/Tanh) to params. Defaults to True.
        as_matrix (bool): If True, return 3x3 matrices, else return 6D vector. Defaults to False.

    Returns:
        Tensor: Transformed pose, shape [..., 6] or [..., 3, 3].
    """
    if pose_tensor.shape[-1] != 6:
        raise ValueError(f"Input pose_tensor must have 6 dimensions in the last axis, got {pose_tensor.shape[-1]}")

    # 确保在正确的设备上操作
    device = pose_tensor.device
    dtype = pose_tensor.dtype

    # 1. 分裂参数
    # split(tensor, split_size_or_sections, dim=-1)
    params = torch.split(pose_tensor, 1, dim=-1)
    scale_x, scale_y, theta, shear, trans_x, trans_y = params[0], params[1], params[2], params[3], params[4], params[5]

    # 2. 应用非线性
    if nonlinear:
        # SELU 对 scale (加偏移量避免非正值)
        # 注意：PyTorch 的 SELU 在 0 附近不完全等于 TF 的 + 1e-2
        # 可以用 softplus 代替，或者直接 SELU 后 clamp
        scale_x = F.selu(scale_x) + 1.0 + 1e-2 # 加 1.0 保证基准为 1 左右
        scale_y = F.selu(scale_y) + 1.0 + 1e-2
        # Tanh 对平移和剪切 (乘以 5)
        trans_x = torch.tanh(trans_x * 5.0)
        trans_y = torch.tanh(trans_y * 5.0)
        shear = torch.tanh(shear * 5.0)
        # 角度转弧度
        theta = theta * (2. * math.pi)
    else:
        # 绝对值 + epsilon (匹配原始代码)
        scale_x = torch.abs(scale_x) + 1e-2
        scale_y = torch.abs(scale_y) + 1e-2

    # 3. 计算三角函数
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # 4. 构建 6 参数 pose 向量 [a, b, tx, c, d, ty]
    if similarity:
        scale = scale_x # 相似变换 x/y scale 相同
        a = scale * cos_theta
        b = -scale * sin_theta
        # tx = trans_x
        c = scale * sin_theta
        d = scale * cos_theta
        # ty = trans_y
        pose_6d = torch.cat([a, b, trans_x, c, d, trans_y], dim=-1)
    else: # 仿射变换
        a = scale_x * cos_theta + shear * scale_y * sin_theta
        b = -scale_x * sin_theta + shear * scale_y * cos_theta
        # tx = trans_x
        c = scale_y * sin_theta
        d = scale_y * cos_theta
        # ty = trans_y
        pose_6d = torch.cat([a, b, trans_x, c, d, trans_y], dim=-1)

    if not as_matrix:
        return pose_6d

    # --- 5. 转换为 3x3 仿射矩阵 ---
    if n_caps is None or n_votes is None:
        raise ValueError("n_caps and n_votes must be provided when as_matrix=True")

    # 获取原始形状前缀 (例如 [B])
    original_shape_prefix = pose_tensor.shape[:-1] # 获取 pose_tensor 中除了最后一维之外的所有维度
    expected_rank = len(original_shape_prefix) # 预期前缀的秩

    # 检查 pose_6d 是否与预期匹配 (... x 6)
    if pose_6d.shape[:-1] != original_shape_prefix:
         # 如果不匹配，尝试基于 B, n_caps, n_votes reshape (需要输入本身形状是对的)
         try:
             batch_size = pose_tensor.shape[0] # 假设第一维是 B
             expected_shape = (batch_size, n_caps, n_votes, 6)
             pose_6d = pose_6d.view(expected_shape) # 尝试 reshape
             original_shape_prefix = pose_6d.shape[:-1] # 更新前缀
             print(f"Reshaped pose_6d to {expected_shape}")
         except RuntimeError as e:
             raise ValueError(f"Could not reshape pose_tensor with original shape {pose_tensor.shape} "
                              f"to match expected logic involving n_caps={n_caps} and n_votes={n_votes}. "
                              f"Error: {e}")

    # 将 pose_6d 展平成 [?, 6] 以方便处理
    pose_6d_flat = pose_6d.reshape(-1, 6)

    # 从 pose_6d_flat 中提取 a, b, tx, c, d, ty
    a, b, tx, c, d, ty = torch.split(pose_6d_flat, 1, dim=-1) # 每个形状 [?, 1]

    # 构建 2x3 部分 [[a, b, tx], [c, d, ty]]
    # 需要移除多余的维度
    zero = torch.zeros_like(a) # 用于占位
    matrix_2x3 = torch.cat([a, b, tx, c, d, ty], dim=-1).view(-1, 2, 3) #[?, 2, 3]

    # 创建最后一行 [0, 0, 1]
    # 需要与 matrix_2x3 的批次维度匹配
    num_matrices = matrix_2x3.shape[0]
    zeros = torch.zeros((num_matrices, 1, 1), device=device, dtype=dtype)
    ones = torch.ones((num_matrices, 1, 1), device=device, dtype=dtype)
    last_row = torch.cat([zeros, zeros, ones], dim=-1) # [?, 1, 3]

    # 拼接得到 3x3 矩阵
    matrix_3x3_flat = torch.cat([matrix_2x3, last_row], dim=-2) # [?, 3, 3]

    # 恢复原始形状前缀
    # original_shape_prefix 应该是 (B, n_caps, n_votes)
    final_shape = original_shape_prefix + (3, 3)
    matrix_3x3 = matrix_3x3_flat.view(final_shape) # [B, n_caps, n_votes, 3, 3]

    return matrix_3x3

# --- 同时迁移 safe_log ---
def safe_log_torch(tensor: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """ PyTorch version of safe_log using clamp """
    safe_tensor = torch.clamp(tensor, min=eps)
    return torch.log(safe_tensor)

# --- 你可以在这里继续添加 relu1_torch, safe_ce_torch, normalize_torch 等函数的实现 ---
def relu1_torch(x: torch.Tensor) -> torch.Tensor:
    """ PyTorch version of relu1 """
    return torch.clamp(F.relu(x), max=1.0) # 或者 F.relu6(x) / 6.0

def normalize_torch(tensor: torch.Tensor, axis: int, eps: float = 1e-8) -> torch.Tensor:
    """ PyTorch version of normalize (L1 norm along axis) """
    return tensor / (torch.sum(tensor, dim=axis, keepdim=True) + eps)

def safe_ce_torch(labels: torch.Tensor, probs: torch.Tensor, axis: int = -1, eps: float = 1e-16) -> torch.Tensor:
    """ PyTorch version of safe_ce (mean cross-entropy) """
    safe_probs = safe_log_torch(probs, eps)
    return torch.mean(-torch.sum(labels * safe_probs, dim=axis))

