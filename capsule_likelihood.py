# geometry.capsule_likelihood.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from typing import Optional, List
from torch_geometric.utils import scatter
import traceback

# 直接使用绝对导入，假设 TuNN 是项目根目录，且运行命令时在此目录
from repr.mlp import MLP
from geometry.geo_torch import safe_log_torch

AttrDict = dict

CapsuleLikelihoodOutputTuple = collections.namedtuple(
    'CapsuleLikelihoodOutputTuple',
    [
        'prediction',           # 最终的标量预测值 [B, 1] or [B]
        'log_prob',             # 批次的平均对数似然 (用于损失)
        # --- 以下为可选，根据需要添加 ---
        'posterior_pre',        # 每个样本的对数似然 [B]
        'vote_presence',        # bool 张量 [B, n_caps, n_votes]?
        'winner',               # hard assignment 结果 [B, n_points, n_dims]?
        'winner_pres',          # winner 的存在概率 [B, n_points]?
    ]
)

class CapsuleLikelihoodTorch(nn.Module):
    def __init__(self,
                 raw_caps_params_dim: int,  # CapsuleLayer 输出的 raw_caps_params 维度
                 n_caps: int,              # Capsule 数量
                 pdf: str = 'normal',
                 pred_mlp1_output_dim: int = 16,
                 pred_mlp1_activation: nn.Module = nn.SELU(),
                 pred_mlp2_hidden_dim: int = 64,
                 pred_mlp2_output_dim: int = 1,
                 pred_mlp2_activation: nn.Module = nn.SELU(),
                 final_activation: nn.Module = nn.ReLU(),
                 use_internal_prediction: bool = False  # 添加此参数
                 ):
        super().__init__()
        if pdf not in ['normal', 'student']:
            raise ValueError("pdf must be 'normal' or 'student'")
        self._pdf = pdf
        self._n_caps = n_caps
        self.use_internal_prediction = use_internal_prediction  # 初始化属性

        # ---- 似然计算相关 ----
        if self._pdf == 'student':
            self.log_df_minus_2 = nn.Parameter(torch.tensor(1.84, dtype=torch.float32))

        # 初始化 MLP 模块
        self.pred_mlp1 = nn.Sequential(
            nn.Linear(raw_caps_params_dim, pred_mlp1_output_dim, bias=True),
            pred_mlp1_activation
        )
        self.pred_mlp2 = nn.Sequential(
            nn.Linear(n_caps, pred_mlp2_hidden_dim, bias=True),
            pred_mlp2_activation,
            nn.Linear(pred_mlp2_hidden_dim, pred_mlp2_output_dim, bias=True),
            final_activation
        )
        self.eps = 1e-10

    def _get_pdf(self, mean: torch.Tensor, scale: torch.Tensor) -> torch.distributions.Distribution:
        """创建概率分布对象"""
        safe_scale = torch.clamp(scale, min=self.eps)
        if self._pdf == 'normal':
            return torch.distributions.Normal(loc=mean, scale=safe_scale)
        elif self._pdf == 'student':
            df = F.softplus(self.log_df_minus_2) + 2.0
            return torch.distributions.StudentT(df=df, loc=mean, scale=safe_scale)
        else: raise ValueError(f"Unsupported PDF type: {self._pdf}")

    def forward(self, x: torch.Tensor, # <-- 目标数据, 预期 [N, 6]
                      caps_layer_output: dict,
                      batch: torch.Tensor, # <-- 批次索引, 预期 [N]
                      presence: Optional[torch.Tensor] = None # <-- 可选 mask, 预期 [N] or [N, 1]
                      ) -> CapsuleLikelihoodOutputTuple:

        # --- 提取必要的张量 ---
        vote_6d = caps_layer_output.get('vote_6d')              # [B_caps, C_obj, V_obj, 6]
        scales = caps_layer_output.get('scale')                 # [B_caps, C_obj, V_obj]
        log_pres_per_vote = caps_layer_output.get('vote_presence_logit') # [B_caps, C_obj, V_obj] (log prob)
        raw_caps_params = caps_layer_output.get('raw_caps_params')    # [B_caps, C_obj, n_params] (用于可选预测)

        # --- 严格检查必需的似然计算输入 ---
        if vote_6d is None or scales is None or log_pres_per_vote is None:
             raise ValueError("Missing required keys ('vote_6d', 'scale', 'vote_presence_logit') in caps_layer_output for likelihood calculation.")

        # --- 获取维度 (如果 x 无效，后面计算会出错，但这里不再提前退出) ---
        N = x.shape[0]
        batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 0

        # --- 获取批次大小 ---
        batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        _B_caps, _n_caps_obj, _n_votes_obj, _dims_6 = vote_6d.shape # B from vote_6d

        # --- 1. 独立预测路径 (默认不执行) ---
        prediction = None
        if self.use_internal_prediction and raw_caps_params is not None and self.pred_mlp1 is not None and self.pred_mlp2 is not None:
            # ---- 这里需要实现完整的独立预测逻辑 ----
            # 例如：
            # if raw_caps_params.shape[0] == batch_size: # 确保批次大小匹配
            #     raw_caps_proc = self.pred_mlp1(raw_caps_params) # [B, C, 16]
            #     # ... (可能需要聚合或其他处理) ...
            #     # final_pred_input = ... # 准备 pred_mlp2 的输入 [B, final_mlp_input_dim]
            #     # prediction = self.pred_mlp2(final_pred_input) # [B, 1]
            # else:
            #      print("Warning: Batch size mismatch for internal prediction path.")
            pass # 暂时不实现内部预测

        # --- 2. 计算似然 ---
        log_prob_batch = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        log_prob_per_example = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        # --- [修改] 检查必要输入是否存在 ---
        if vote_6d is None or scales is None or log_pres_per_vote is None or batch_size == 0:
            print("Warning: Missing required inputs for likelihood or empty batch. Skipping calculation.")

        elif vote_6d.shape[0] != batch_size: # 检查批次大小匹配
            print(f"Warning: Batch size mismatch between vote_6d ({vote_6d.shape[0]}) and input data ({batch_size}). Skipping likelihood calculation.")
        else:
            try: # 保持 try-except
                # --- 广播参数 ---
                votes_6d_exp = vote_6d[batch]
                scales_exp = scales[batch]
                mixing_log_prior_exp = log_pres_per_vote[batch]

                # --- 准备 PDF 计算所需的形状 (需要确保 x 是 [N, 6]) ---
                if x.shape[1] != 6: # 添加运行时的检查
                    raise ValueError(f"Likelihood calculation requires input x dim 1 to be 6, got {x.shape[1]}")
                expanded_x_for_prob = x.unsqueeze(1).unsqueeze(2)
                expanded_scales_for_prob = scales_exp.unsqueeze(-1)

                # --- 计算高斯/StudentT 对数似然 ---
                vote_component_pdf = self._get_pdf(votes_6d_exp, expanded_scales_for_prob)
                # 计算 log_prob: log_prob(x | mean=vote, scale=scale)
                vote_log_prob_per_dim = vote_component_pdf.log_prob(expanded_x_for_prob) # [N, C_obj, V_obj, 6]
                # 对特征维度求和
                vote_log_prob = torch.sum(vote_log_prob_per_dim, dim=-1) # [N, C_obj, V_obj]

                # --- 计算混合模型的对数似然 ---
                # log( sum_{c,v} [ p(v|c)*p(c) * p(x|v,c) ] )
                # mixing_log_prior_exp 对应 log(p(v|c)*p(c)) or log(p(v,c))
                posterior_logits = mixing_log_prior_exp + vote_log_prob # [N, C_obj, V_obj]

                # 计算每个数据点 x_i 的边缘对数似然 log p(x_i)
                # 使用 logsumexp 技巧在胶囊 C 和投票 V 维度上求和
                log_prob_per_point = torch.logsumexp(posterior_logits, dim=(1, 2)) # [N]

                # --- 应用 presence mask (如果提供) ---
                if presence is not None:
                    if presence.shape[0] != N:
                        print(f"Warning: Presence mask shape {presence.shape} mismatch with N={N}. Ignoring presence mask.")
                    else:
                        log_prob_per_point = log_prob_per_point * presence.view(-1).float()

                # --- 聚合得到每个图的对数似然 ---
                log_prob_per_example = scatter(log_prob_per_point, batch, dim=0, reduce='sum', dim_size=batch_size) # [B]

                # --- 计算批次平均对数似然 ---
                log_prob_batch = torch.mean(log_prob_per_example) # scalar

            except Exception as e:
                print(f"ERROR during likelihood calculation: {e}")
                traceback.print_exc()
                # 出错时保持 log_prob 为 0
                log_prob_batch = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                log_prob_per_example = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        # --- 其他输出 (占位符) ---
        vote_presence = None
        winner = None
        winner_pres = None

        return CapsuleLikelihoodOutputTuple(
            prediction=prediction,           # 当前为 None
            log_prob=log_prob_batch,         # scalar
            posterior_pre=log_prob_per_example, # [B]
            vote_presence=None,              # Placeholder: None
            winner=None,                     # Placeholder: None
            winner_pres=None                 # Placeholder: None
        )
    
if __name__ == "__main__":
    # 测试 CapsuleLikelihoodTorch 的功能
    print("Running CapsuleLikelihoodTorch internal test...")

    # 模拟输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 10  # 节点数量
    B = 2   # 批次大小
    n_caps = 5  # Capsule 数量
    n_votes = 3  # 每个 Capsule 的投票数
    raw_caps_params_dim = 16  # Capsule 原始参数维度

    # 模拟输入张量
    x = torch.randn(N, 6, device=device)  # 输入数据 [N, 6]
    batch = torch.tensor([0] * 5 + [1] * 5, device=device)  # 批次索引 [N]
    presence = torch.randint(0, 2, (N,), device=device).float()  # Presence mask [N]

    # 模拟 Capsule 层输出
    caps_layer_output = {
        'vote_6d': torch.randn(B, n_caps, n_votes, 6, device=device),  # [B, n_caps, n_votes, 6]
        'scale': torch.rand(B, n_caps, n_votes, device=device),        # [B, n_caps, n_votes]
        'vote_presence_logit': torch.randn(B, n_caps, n_votes, device=device),  # [B, n_caps, n_votes]
        'raw_caps_params': torch.randn(B, n_caps, raw_caps_params_dim, device=device)  # [B, n_caps, raw_caps_params_dim]
    }

    # 实例化 CapsuleLikelihoodTorch
    capsule_likelihood = CapsuleLikelihoodTorch(
        raw_caps_params_dim=raw_caps_params_dim,
        n_caps=n_caps,
        pdf='normal',  # 使用正态分布
        pred_mlp1_output_dim=16,
        pred_mlp1_activation=torch.nn.SELU(),
        pred_mlp2_hidden_dim=64,
        pred_mlp2_output_dim=1,
        pred_mlp2_activation=torch.nn.SELU(),
        final_activation=torch.nn.ReLU()
    ).to(device)

    # 前向传播
    try:
        output = capsule_likelihood(x, caps_layer_output, batch, presence)

        # 验证输出
        assert isinstance(output, CapsuleLikelihoodOutputTuple), "Output is not of type CapsuleLikelihoodOutputTuple"
        assert output.log_prob.dim() == 0, "log_prob should be a scalar"
        assert output.posterior_pre.shape == (B,), f"posterior_pre shape mismatch: {output.posterior_pre.shape}"

        print("Test passed!")
        print(f"Log probability (batch): {output.log_prob.item()}")
        print(f"Log probability (per example): {output.posterior_pre}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()