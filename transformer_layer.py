# repr.transfomer_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import List, Tuple, Optional, Dict

# 导入依赖 (使用绝对导入)
try:
    from repr.mlp import MLP
    from repr.stoi import StoiRepTorch
    from torch_geometric.nn.aggr import Aggregation # 导入 Aggregation 基类或具体的聚合器
except ImportError as e:
    print(f"Error importing dependencies for TransformerLayerTorch: {e}")
    raise

class TransformerLayerTorch(nn.Module):
    """
    PyTorch 版本的 atom_che_env (Graph Network 更新层)。
    包含 Stoi_Rep 调用和节点、边、全局状态的更新。
    使用 MLP 替代 phi_e, phi_v, phi_u 中的 Conv2D+Attention。
    """
# modules/transformer_layer.py
class TransformerLayerTorch(nn.Module):
    def __init__(self,
                 v_dim: int, e_dim: int, u_dim: int,
                 hidden_units_v: List[int] = [128, 128],
                 hidden_units_e: List[int] = [128, 128],
                 hidden_units_u: List[int] = [128, 128],
                 stoi_config: dict = {},
                 pool_method: str = "mean", # <-- 接收 pool_method
                 activation: nn.Module = nn.SELU(),
                 ):
        super().__init__()
        # [--- BEGIN MODIFICATION ---]
        # --- 存储 pool_method 到 self.reduce_op ---
        if pool_method not in ["mean", "sum"]:
            raise ValueError("pool_method must be 'mean' or 'sum'")
        self.pool_method = pool_method # 可以保留
        self.reduce_op = pool_method   # <-- 添加这一行来存储
        # [--- END MODIFICATION ---]
        self.activation = activation
        self.v_dim, self.e_dim, self.u_dim = v_dim, e_dim, u_dim

        # ... (实例化 StoiRepTorch) ...
        self.stoi_rep = StoiRepTorch(**stoi_config)

        # print(f"Initializing MLPs in TransformerLayerTorch with v_dim={v_dim}, e_dim={e_dim}, u_dim={u_dim}")

        # [--- BEGIN MODIFICATION ---]
        # --- 确定 MLP 的输出维度 (必须与输入对应维度相同) ---
        phi_e_output_dim = self.e_dim
        phi_v_output_dim = self.v_dim # <-- 直接使用输入维度
        phi_u_output_dim = self.u_dim # <-- 直接使用输入维度

        # --- 计算 MLP 的输入维度 (使用正确的输出维度) ---
        phi_e_input_dim = 2 * self.v_dim + self.e_dim + self.u_dim
        phi_v_input_dim = phi_e_output_dim + self.v_dim + self.u_dim # 使用 phi_e 的输出 (e_dim)
        phi_u_input_dim = phi_e_output_dim + phi_v_output_dim + self.u_dim # 使用 phi_e 和 phi_v 的输出 (e_dim, v_dim)
        # [--- END MODIFICATION ---]


        # --- 创建 MLP 实例，使用正确的 hidden_dims + output_dim ---
        print(f"\nCreating phi_e_mlp:")
        phi_e_dims = hidden_units_e + [phi_e_output_dim] # 例如 [128, 128, 100]
        print(f"  input_dim={phi_e_input_dim}")
        print(f"  hidden_dims={phi_e_dims}")
        self.phi_e_mlp = MLP(input_dim=phi_e_input_dim, hidden_dims=phi_e_dims,
                             activation=activation, activate_last=True)
        print(f"  phi_e_mlp created: output={phi_e_output_dim}")

        print(f"\nCreating phi_v_mlp:")
        phi_v_dims = hidden_units_v + [phi_v_output_dim] # 例如 [128, 128, 192]
        print(f"  input_dim={phi_v_input_dim}")
        print(f"  hidden_dims={phi_v_dims}")
        self.phi_v_mlp = MLP(input_dim=phi_v_input_dim, hidden_dims=phi_v_dims,
                             activation=activation, activate_last=True)
        print(f"  phi_v_mlp created: output={phi_v_output_dim}")

        print(f"\nCreating phi_u_mlp:")
        phi_u_dims = hidden_units_u + [phi_u_output_dim] # 例如 [128, 128, 192]
        print(f"  input_dim={phi_u_input_dim}")
        print(f"  hidden_dims={phi_u_dims}")
        self.phi_u_mlp = MLP(input_dim=phi_u_input_dim, hidden_dims=phi_u_dims,
                             activation=activation, activate_last=True)
        print(f"  phi_u_mlp created: output={phi_u_output_dim}")



    def forward(self, atom_vec_: torch.Tensor,    # 节点特征 [N, v_dim]
                      bond_vec_: torch.Tensor,    # 边特征 [N_edges, e_dim]
                      state_vec_: torch.Tensor,   # 节点级别状态 [N, u_dim]
                      com_w: torch.Tensor,       # 组成权重 [N, 1]
                      atom_fea: torch.Tensor,    # one-hot 特征 [N, 103]
                      edge_index: torch.Tensor,  # [2, N_edges] (假设 [source, target])
                      batch: torch.Tensor,       # [N]
                      loop_num: int,             # (未使用)
                      batch_size: int,           # 批次大小 B
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一次 Graph Network 更新。
        """
        N = atom_vec_.shape[0]
        N_edges = edge_index.shape[1]
        device = atom_vec_.device

        # ---- 1. 计算化学计量特征 (comp_che) ----
        comp_che = self.stoi_rep(com_w, atom_fea, batch) # [B, comp_fea_dim]

        # ---- 2. 准备全局状态 u ----
        u = scatter(state_vec_, batch, dim=0, reduce=self.reduce_op, dim_size=batch_size) # [B, u_dim]

        # ---- 3. 边更新 (phi_e) ----
        # PyG 约定: edge_index = [source, target]
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        # fs: target node features, fr: source node features
        fs = atom_vec_[target_nodes] # [N_edges, v_dim]
        fr = atom_vec_[source_nodes] # [N_edges, v_dim]

        # 广播全局状态 u 到每条边 (使用 target node 的 batch index)
        bond_batch = batch[target_nodes]
        u_expand_e = u[bond_batch]        # [N_edges, u_dim]

        # 拼接 [target_node, source_node, edge_features, global_state]
        phi_e_input = torch.cat([fs, fr, bond_vec_, u_expand_e], dim=-1)
        bond_che = self.phi_e_mlp(phi_e_input) # [N_edges, e_dim] (输出维度 = e_dim)

        # ---- 4. 聚合到节点 (rho_e_v) ----
        # 将更新后的边特征聚合到 **目标** 节点 (target_nodes)
        # 输出维度 = e_dim
        b_ei_p = scatter(bond_che, target_nodes, dim=0, reduce=self.reduce_op, dim_size=N) # [N, e_dim]

        # ---- 5. 节点更新 (phi_v) ----
        # 广播全局状态 u 到每个节点
        u_expand_v = u[batch] # [N, u_dim]

        # 拼接 [aggregated_edges, node_features, global_state]
        # 输入维度: e_dim + v_dim + u_dim
        phi_v_input = torch.cat([b_ei_p, atom_vec_, u_expand_v], dim=-1)
        atom_che = self.phi_v_mlp(phi_v_input) # [N, v_dim] (输出维度 = v_dim)

        # ---- 6. 聚合到全局 (rho_e_u, rho_v_u) ----
        # 聚合更新后的边特征到全局 (使用 target node 的 batch index?)
        b_e_p = scatter(bond_che, bond_batch, dim=0, reduce=self.reduce_op, dim_size=batch_size) # [B, e_dim]
        # 聚合更新后的节点特征到全局
        b_v_p = scatter(atom_che, batch, dim=0, reduce=self.reduce_op, dim_size=batch_size) # [B, v_dim]

        # ---- 7. 全局更新 (phi_u) ----
        # 拼接 [aggregated_edges, aggregated_nodes, global_state]
        # 输入维度: e_dim + v_dim + u_dim
        phi_u_input = torch.cat([b_e_p, b_v_p, u], dim=-1)
        state_che = self.phi_u_mlp(phi_u_input) # [B, u_dim] (输出维度 = u_dim)

        # ---- 8. 返回结果 ----
        # 广播更新后的全局状态回节点
        state_che_broadcast = state_che[batch] # [N, u_dim]

        # 返回: 更新后节点, 更新后边, 更新后节点级状态, 图级化学计量
        return atom_che, bond_che, state_che_broadcast, comp_che