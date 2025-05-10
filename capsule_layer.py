# geometry.capsule_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from typing import List, Optional

from repr.batch_mlp_torch import BatchMLPTorch
from geometry.geo_torch import geometric_transform_torch, safe_log_torch

AttrDict = dict # 使用普通字典

class CapsuleLayerTorch(nn.Module):
    """PyTorch 版 CapsuleLayer (OCAE block)"""

    _n_transform_params = 6

    def __init__(self, n_caps: int, n_caps_dims: int, n_votes: int,
                 n_caps_params: Optional[int] = None,
                 n_hiddens: List[int] = [128],
                 learn_vote_scale: bool = False,
                 deformations: bool = True,
                 noise_type: Optional[str] = None,
                 noise_scale: float = 0.,
                 similarity_transform: bool = True,
                 caps_dropout_rate: float = 0.0,
                 activation: nn.Module = nn.SELU()):
        super().__init__()
        # ... (__init__ 代码与上次一致，创建 MLPs, Parameters, Biases) ...
        if n_caps_dims != 2:
             print(f"Warning: Using n_caps_dims={n_caps_dims}. Geometric transform assumes 2D pose params.")

        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_caps_params = n_caps_params
        self._n_votes = n_votes
        self._n_hiddens = n_hiddens if isinstance(n_hiddens, list) else [n_hiddens]
        self._learn_vote_scale = learn_vote_scale
        self._deformations = deformations
        self._noise_type = noise_type
        self._noise_scale = noise_scale
        self._similarity_transform = similarity_transform
        self._caps_dropout_rate = caps_dropout_rate
        self.activation = activation

        # 1. 参数预测 MLP (如果需要)
        self.param_predict_mlp = None
        if self._n_caps_params is not None:
            self.param_predict_mlp = BatchMLPTorch(
                n_hiddens=self._n_hiddens + [self._n_caps * self._n_caps_params],
                activation=self.activation,
                activate_final=False
            )

        # 2. 计算输出参数总维度和分裂点
        self.output_shapes = collections.OrderedDict({
            'cpr_dynamic': (self._n_votes, self._n_transform_params),
            'ccr': (1, self._n_transform_params),
            'pres_logit_per_caps': (1,),
            'pres_logit_per_vote': (self._n_votes,),
            'scale_per_vote': (self._n_votes,)
        })
        self.splits = [np.prod(s).item() for s in self.output_shapes.values()]
        n_outputs = sum(self.splits)

        # 3. Capsule 参数 -> 输出参数的 MLP
        #    输入维度将在 forward 中确定
        self.caps_mlp = BatchMLPTorch(
            input_dim = None, # Lazy initialization
            n_hiddens=[h for h in self._n_hiddens] + [n_outputs],
            activation=self.activation,
            activate_final=False,
            use_bias=False
        )

        # 4. 可学习的静态变换参数 cpr_static
        self.cpr_static = nn.Parameter(torch.zeros(1, self._n_caps, self._n_votes, self._n_transform_params))
        nn.init.normal_(self.cpr_static, std=0.1)

        # 5. 添加偏置参数
        self.bias_ccr = nn.Parameter(torch.zeros(1, self._n_caps, 1, self._n_transform_params))
        self.bias_pres_logit_per_caps = nn.Parameter(torch.zeros(1, self._n_caps, 1))
        self.bias_pres_logit_per_vote = nn.Parameter(torch.zeros(1, self._n_caps, self._n_votes))
        self.bias_scale_per_vote = nn.Parameter(torch.zeros(1, self._n_caps, self._n_votes))


    def forward(self, features: torch.Tensor,
                      parent_transform: Optional[torch.Tensor] = None,
                      parent_presence: Optional[torch.Tensor] = None
                      ) -> dict:

        batch_size = features.shape[0]
        device = features.device
        dtype = features.dtype
        
        # print(f"  DEBUG (CapsuleLayer Entry): features requires_grad = {features.requires_grad}")

        # 1. 预测或获取 Capsule 参数
        raw_caps_params = None
        if self.param_predict_mlp is not None:
            raw_caps_params_flat = self.param_predict_mlp(features)
            # if self._n_caps_params is None or self._n_caps_params <= 0:
                 # raise ValueError("n_caps_params must be positive if param_predict_mlp is used.")
            # try:
            caps_params = raw_caps_params_flat.view(batch_size, self._n_caps, self._n_caps_params)
            # except RuntimeError as e:
                 # raise RuntimeError(f"Failed to reshape raw_caps_params_flat {raw_caps_params_flat.shape} "
                                  # f"to [B={batch_size}, n_caps={self._n_caps}, n_params={self._n_caps_params}]. Error: {e}")
            raw_caps_params = caps_params
            # print(f"    DEBUG (CapsuleLayer): caps_params (predicted) requires_grad = {caps_params.requires_grad}") # <-- 检查预测后
        else:
            # if features.dim() < 3 or features.shape[1] != self._n_caps:
            #      raise ValueError(f"Input features shape {features.shape} incompatible when n_caps_params is None.")
            caps_params = features
            # if self._n_caps_params is None: self._n_caps_params = features.shape[2]
            # elif self._n_caps_params != features.shape[2]: raise ValueError("Input feature dim changed!")
            raw_caps_params = caps_params
            # print(f"    DEBUG (CapsuleLayer): caps_params (direct) requires_grad = {caps_params.requires_grad}") # <-- 检查直接使用时

        # 2. Capsule Dropout
        if self._caps_dropout_rate > 0.0 and self.training:
            keep_prob = 1.0 - self._caps_dropout_rate
            caps_exist = torch.bernoulli(torch.full((batch_size, self._n_caps, 1), keep_prob,
                                                    device=device, dtype=dtype))
            # print(f"    DEBUG (CapsuleLayer): caps_exist (dropout) requires_grad = {caps_exist.requires_grad}") # <-- 检查 dropout 后
        else:
            caps_exist = torch.ones((batch_size, self._n_caps, 1), device=device, dtype=dtype)
            # print(f"    DEBUG (CapsuleLayer): caps_exist (ones) requires_grad = {caps_exist.requires_grad}") # <-- 检查 ones

        # 3. 准备 caps_mlp 的输入
        if caps_params is None: raise RuntimeError("caps_params is None")
        # expected_caps_mlp_input_dim = caps_params.shape[-1] + 1
        # if not self.caps_mlp._initialized:
        #     self.caps_mlp._input_dim = expected_caps_mlp_input_dim
        #     self.caps_mlp._build_network(expected_caps_mlp_input_dim)
        #     self.caps_mlp = self.caps_mlp.to(device)
        # elif self.caps_mlp._input_dim != expected_caps_mlp_input_dim:
        #      raise ValueError(f"caps_mlp expected input dim {self.caps_mlp._input_dim} but got {expected_caps_mlp_input_dim}")
        caps_mlp_input = torch.cat([caps_params, caps_exist], dim=-1)
        # print(f"    DEBUG (CapsuleLayer): caps_mlp_input requires_grad = {caps_mlp_input.requires_grad}") # <-- 检查 cat 之后

        # 4. 预测输出参数
        all_params = self.caps_mlp(caps_mlp_input)
        # print(f"    DEBUG (CapsuleLayer): all_params requires_grad = {all_params.requires_grad}") # <-- 检查 MLP 输出

        # 5. 分裂参数
        split_params_list = torch.split(all_params, self.splits, dim=-1)
        params_dict = {}
        for i, key in enumerate(self.output_shapes.keys()):
             params_dict[key] = split_params_list[i].view((batch_size, self._n_caps) + self.output_shapes[key])

        cpr_dynamic = params_dict['cpr_dynamic'] # [B, n_caps, n_votes, 6]
        # print(f"    DEBUG (CapsuleLayer): cpr_dynamic requires_grad (after split) = {cpr_dynamic.requires_grad}") # <-- 检查分裂后

        ccr_params = params_dict['ccr']
        pres_logit_per_caps_params = params_dict['pres_logit_per_caps']
        pres_logit_per_vote_params = params_dict['pres_logit_per_vote']
        scale_per_vote_params = params_dict['scale_per_vote']

        # 6. 添加偏置
        ccr = ccr_params + self.bias_ccr
        pres_logit_per_caps = pres_logit_per_caps_params + self.bias_pres_logit_per_caps
        pres_logit_per_vote = pres_logit_per_vote_params + self.bias_pres_logit_per_vote
        scale_per_vote = scale_per_vote_params + self.bias_scale_per_vote

        # 7. 应用 Capsule Dropout 影响
        if self._caps_dropout_rate > 0.0:
             pres_logit_per_caps = pres_logit_per_caps + safe_log_torch(caps_exist, eps=1e-9)

        # 8. 添加噪声 (省略)

        # 9. 计算变换矩阵
        if parent_transform is None:
            ccr_matrix = geometric_transform_torch(ccr.squeeze(2),
                                                   n_caps=self._n_caps, n_votes=1,
                                                   similarity=self._similarity_transform,
                                                   nonlinear=True, as_matrix=True)
        else:
            if parent_transform.shape != (batch_size, self._n_caps, 3, 3):
                 raise ValueError(f"Shape mismatch for parent_transform: expected [...] got {parent_transform.shape}")
            ccr_matrix = parent_transform

        cpr_params = cpr_dynamic + self.cpr_static
        # --- [修改结束] ---

        cpr_matrix = geometric_transform_torch(cpr_params,
                                               n_caps=self._n_caps, n_votes=self._n_votes,
                                               similarity=self._similarity_transform,
                                               nonlinear=True, as_matrix=True)

        # 10. 计算投票
        ccr_per_vote_matrix = ccr_matrix.unsqueeze(2).expand(-1, -1, self._n_votes, -1, -1)
        votes = torch.matmul(ccr_per_vote_matrix, cpr_matrix)
        top_2x3 = votes[..., :2, :]
        vote_6d = top_2x3.reshape(batch_size, self._n_caps, self._n_votes, 6)

        # 11. 计算存在概率
        if parent_presence is not None:
             log_pres_per_caps = safe_log_torch(parent_presence.view(batch_size, self._n_caps, 1), eps=1e-9)
        else:
             log_pres_per_caps = F.logsigmoid(pres_logit_per_caps)
        log_pres_per_vote_cond = F.logsigmoid(pres_logit_per_vote)
        log_pres_per_vote = log_pres_per_caps + log_pres_per_vote_cond
        object_caps_presence_prob = torch.sigmoid(pres_logit_per_caps.squeeze(-1))

        # 12. 计算尺度
        if self._learn_vote_scale:
            scale_per_vote = F.softplus(scale_per_vote + 0.5) + 1e-2
        else:
            scale_per_vote = torch.ones_like(scale_per_vote)

        # 13. 计算 L2 损失值
        dynamic_weights_l2 = torch.sum(cpr_dynamic**2) * 0.5 / float(batch_size)
        # print(f"  DEBUG (CapsuleLayer): Calculated dynamic_weights_l2 = {dynamic_weights_l2.item():.4e} (requires_grad={dynamic_weights_l2.requires_grad})")

        # 14. 返回结果字典
        return AttrDict(
            vote=votes,
            vote_6d=vote_6d,
            scale=scale_per_vote,
            vote_presence_logit=log_pres_per_vote,
            pres_logit_per_caps=pres_logit_per_caps,
            pres_logit_per_vote=pres_logit_per_vote,
            object_caps_presence_prob=object_caps_presence_prob,
            dynamic_weights_l2=dynamic_weights_l2,
            raw_caps_params=raw_caps_params,
        )