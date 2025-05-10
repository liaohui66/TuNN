# matchecon.py (再次修正，移除 IrrepsArray)

import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn
from e3nn import o3
# from e3nn import IrrepsArray # REMOVED
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct, Norm, Irreps, SphericalHarmonics
from e3nn.nn import Gate, NormActivation
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, radius_graph
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax
from typing import List, Tuple, Optional

# --- RadialMLP (No changes needed, operates on standard tensors) ---
class RadialMLP(nn.Module):
    # ... (no changes) ...
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.SiLU):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x.float())
    
# def check_clebsch_gordan(ir_in1: o3.Irrep, ir_in2: o3.Irrep, ir_out: o3.Irrep) -> bool:
#     """检查 ir_out 是否包含在 ir_in1 x ir_in2 的张量积中"""
#     l_rule = abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l
#     p_rule = ir_out.p == ir_in1.p * ir_in2.p
#     return l_rule and p_rule


# --- EdgeKeyValueNetwork (应用了 TensorProduct 替换) ---
class EdgeKeyValueNetwork(nn.Module):
    def __init__(self, irreps_node_input, irreps_sh, irreps_key_output, irreps_value_output, num_basis_radial, radial_mlp_hidden):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_key_output = o3.Irreps(irreps_key_output)
        self.irreps_value_output = o3.Irreps(irreps_value_output)
        self.num_basis_radial = num_basis_radial

        self.tp_k = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_key_output,
            shared_weights=False
        )
        self.fc_k = RadialMLP(self.num_basis_radial, radial_mlp_hidden, self.tp_k.weight_numel)

        self.tp_v = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_value_output,
            shared_weights=False
        )
        self.fc_v = RadialMLP(self.num_basis_radial, radial_mlp_hidden, self.tp_v.weight_numel)

    def forward(self, node_features_src: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_weights = self.fc_k(edge_radial_emb).to(node_features_src.device)
        k_on_edge = self.tp_k(node_features_src, edge_sh, weight=k_weights)
        v_weights = self.fc_v(edge_radial_emb).to(node_features_src.device)
        v_on_edge = self.tp_v(node_features_src, edge_sh, weight=v_weights)
        return k_on_edge, v_on_edge

# --- TFN Interaction Block (应用了 TensorProduct 替换) ---
class TFNInteractionBlock(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, irreps_sh, num_basis_radial, radial_mlp_hidden=[64], activation_gate=True):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_basis_radial = num_basis_radial

        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_node_output,
            shared_weights=False
        )
        self.radial_to_tp_weights = RadialMLP(num_basis_radial, radial_mlp_hidden, self.tensor_product.weight_numel)

        self.linear = Linear(self.irreps_node_output, self.irreps_node_output)

        target_irreps = self.irreps_node_output
        act_norm = F.silu
        self.activation = NormActivation(target_irreps, act_norm)

        if self.irreps_node_input != self.irreps_node_output:
            self.skip_connection_project = Linear(self.irreps_node_input, self.irreps_node_output, internal_weights=True)
        else:
            self.skip_connection_project = None
        # self.activation = nn.Identity()
        # self.skip_connection_project = None

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor,
                edge_length: Optional[torch.Tensor] = None,
                max_radius: Optional[float] = None
                ) -> torch.Tensor:
        # ... (省略 forward 代码, 确保 self.tensor_product 调用时使用 weights=tp_weights) ...
        edge_src, edge_dst = edge_index
        tp_weights = self.radial_to_tp_weights(edge_radial_emb).to(node_features.device)
        node_features_src = node_features[edge_src]
        # 调用 FullTensorProduct 的 forward 时，同样使用 weights 参数
        messages = self.tensor_product(node_features_src, edge_sh, weight=tp_weights)
        aggregated_messages = scatter_sum(messages, edge_dst, dim=0, dim_size=node_features.shape[0])
        transformed_messages = self.linear(aggregated_messages)
        activated_messages = self.activation(transformed_messages)
        if self.skip_connection_project is not None:
            skip_features = self.skip_connection_project(node_features)
        else:
            skip_features = node_features
        updated_node_features = skip_features + activated_messages
        return updated_node_features

        # updated_node_features = activated_messages
        # return updated_node_features
    
# --- SE(3) Transformer Interaction Block (Adjusted for Tensors) ---
class SE3TransformerInteractionBlock(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, irreps_sh, num_basis_radial, radial_mlp_hidden=[64], num_attn_heads=4, fc_neurons=[128, 128], activation_gate=True, use_layer_norm=False):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_basis_radial = num_basis_radial
        self.num_attn_heads = num_attn_heads
        self.use_layer_norm = use_layer_norm

        # --- 1. Define Irreps for Q, K, V per head ---
        assert all(mul % self.num_attn_heads == 0 for mul, ir in self.irreps_node_input), f"Input irreps channels {self.irreps_node_input} must be divisible by num_attn_heads={self.num_attn_heads}"
        self.irreps_value_head = o3.Irreps([ (mul // self.num_attn_heads, ir) for mul, ir in self.irreps_node_input ])
        self.irreps_value = self.irreps_value_head * self.num_attn_heads
        self.irreps_key_query_head = self.irreps_value_head # Using same structure
        self.irreps_key_query = self.irreps_value

        # --- 2. Networks for Q, K, V ---
        self.query_network = Linear(self.irreps_node_input, self.irreps_key_query, internal_weights=True)
        self.kv_network = EdgeKeyValueNetwork(
            irreps_node_input=self.irreps_node_input, irreps_sh=self.irreps_sh,
            irreps_key_output=self.irreps_key_query, irreps_value_output=self.irreps_value,
            num_basis_radial=self.num_basis_radial, radial_mlp_hidden=radial_mlp_hidden
        )

        # --- 3. Attention Mechanism ---
        # Need TP for single head dot product
        self.dot_product_single_head = o3.FullyConnectedTensorProduct(
            self.irreps_key_query_head, self.irreps_key_query_head, "0e"
        )
        self.scale = 1.0 / (self.irreps_key_query_head.dim ** 0.5)

        # --- 4. Output Projection ---
        self.output_projection = Linear(self.irreps_value, self.irreps_node_output, internal_weights=True)

        # --- 5. Feed-Forward Network (FFN) ---
        # Gate initialization logic (already corrected)
        ffn_intermediate_irreps = []
        ffn_act_scalars, ffn_act_gates, ffn_act_features = [], [], []
        ffn_irreps_scalars, ffn_irreps_gated = o3.Irreps(""), o3.Irreps("")
        for mul, ir in self.irreps_node_output: # Base expansion on output irreps
             if ir.l == 0: ffn_intermediate_irreps.append((mul * 2, ir))
             else: ffn_intermediate_irreps.append((mul, ir))
        ffn_intermediate_irreps = o3.Irreps(ffn_intermediate_irreps).simplify()

        target_irreps = ffn_intermediate_irreps
        act_norm = F.silu
        self.ffn_activation = NormActivation(target_irreps, act_norm) # Pass the corrected list

        self.ffn = nn.Sequential(
            Linear(self.irreps_node_output, ffn_intermediate_irreps, internal_weights=True),
            self.ffn_activation,
            Linear(ffn_intermediate_irreps, self.irreps_node_output, internal_weights=True)
        )

        # --- 6. Skip connection projection (if needed) ---
        if self.irreps_node_input != self.irreps_node_output:
            self.skip_projection_attn = Linear(self.irreps_node_input, self.irreps_node_output, internal_weights=True)
        else:
            self.skip_projection_attn = None

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor, edge_length: Optional[torch.Tensor] = None, max_radius: Optional[float] = None) -> torch.Tensor: # Input/Output Tensors
        N = node_features.shape[0]
        E = edge_index.shape[1]
        edge_src, edge_dst = edge_index
        H = self.num_attn_heads

        # --- 1. Calculate Q, K, V ---
        q = self.query_network(node_features) # Tensor [N, K_dim]
        k_on_edge, v_on_edge = self.kv_network(node_features[edge_src], edge_sh, edge_radial_emb) # Tensors [E, K_dim], [E, V_dim]

        # --- 2. Reshape for Multi-Head Attention ---
        q_heads = q.reshape(N, H, self.irreps_key_query_head.dim) # [N, H, kq_head_dim]
        k_heads = k_on_edge.reshape(E, H, self.irreps_key_query_head.dim) # [E, H, kq_head_dim]
        v_heads = v_on_edge.reshape(E, H, self.irreps_value_head.dim)   # [E, H, v_head_dim]

        # --- 3. Calculate Attention Scores ---
        q_heads_on_edge = q_heads[edge_dst] # [E, H, kq_head_dim]

        # Calculate dot product per head
        attn_logits_list = []
        for h in range(H):
            # TP expects tensors, Irreps are known by self.dot_product_single_head
            q_h = q_heads_on_edge[:, h, :] # [E, kq_head_dim]
            k_h = k_heads[:, h, :]         # [E, kq_head_dim]
            dot_h = self.dot_product_single_head(q_h, k_h) # Output [E, 1]
            attn_logits_list.append(dot_h)

        attn_logits = torch.cat(attn_logits_list, dim=-1) * self.scale # [E, H]
        # print(f"attn_logits shape before softmax: {attn_logits.shape}")

        # Optional cutoff
        if edge_length is not None and max_radius is not None:
            # --- Cutoff 逻辑 ---
            def soft_unit_step(x, sharpness=10.0): return torch.sigmoid(sharpness * x)
            assert edge_length.shape[0] == E, f"edge_length shape mismatch: {edge_length.shape}"
            # 确保 edge_length 是 [E] 或 [E, 1]
            edge_weight_cutoff = soft_unit_step(1.0 - edge_length.squeeze(-1) / max_radius) # 先 squeeze 确保是 [E] 再计算
            assert edge_weight_cutoff.shape == (E,), f"edge_weight_cutoff shape mismatch: {edge_weight_cutoff.shape}"
            # 明确 unsqueeze 用于广播
            edge_weight_cutoff = edge_weight_cutoff.unsqueeze(-1).to(attn_logits.device) # 形状变为 [E, 1]

            # 检查 attn_logits 形状是否符合预期
            assert attn_logits.shape == (E, H), f"attn_logits shape before cutoff multiply mismatch: {attn_logits.shape}"

            # 进行广播乘法
            attn_logits = attn_logits * edge_weight_cutoff # [E, H] * [E, 1] -> [E, H]
            # --- Cutoff 逻辑结束 ---
            # print(f"attn_logits shape after cutoff: {attn_logits.shape}")

        # --- 4. Softmax ---
        attn_weights = scatter_softmax(attn_logits.float(), edge_dst, dim=0) # [E, H]
        # print(f"attn_weights shape after softmax: {attn_weights.shape}")

        # --- 5. 聚合加权的 Values ---
        # print(f"v_heads shape before multiply: {v_heads.shape}")
        weighted_v = v_heads * attn_weights.unsqueeze(-1) # [E, H, v_head_dim]

        # 在 SE3TransformerInteractionBlock.forward 中 reshape 之前
        # print(f"--- Attention Debug ---")
        # print(f"E = {E}, H = {H}")
        # 打印来自 EdgeKeyValueNetwork 的原始输出
        # print(f"v_on_edge shape: {v_on_edge.shape}")
        # print(f"v_on_edge numel: {v_on_edge.numel()}")
        # print(f"Expected v_on_edge dim: {self.kv_network.irreps_value_output.dim}") # 正确访问
        # print(f"Expected v_on_edge numel: {E * self.kv_network.irreps_value_output.dim}") # 正确访问

        # 打印 reshape v_heads 之前的信息
        # print(f"irreps_value_head.dim: {self.irreps_value_head.dim}")
        # print(f"Attempting v_heads reshape to: ({E}, {H}, {self.irreps_value_head.dim})")
        v_heads = v_on_edge.reshape(E, H, self.irreps_value_head.dim) # 这步可能也会报错如果v_on_edge numel不对
        # print(f"v_heads shape: {v_heads.shape}")

        # 打印乘法之前的信息
        # print(f"attn_weights shape: {attn_weights.shape}")
        weighted_v = v_heads * attn_weights.unsqueeze(-1)
        # print(f"weighted_v shape: {weighted_v.shape}")
        # print(f"weighted_v numel: {weighted_v.numel()}") # 打印实际元素数量

        # 打印最终 reshape 的信息
        # print(f"Attempting final reshape to: ({E * H}, {self.irreps_value_head.dim})")
        # print(f"Expected final numel: {E * H * self.irreps_value_head.dim}")

        weighted_v_flat = weighted_v.reshape(E * H, self.irreps_value_head.dim) # [E*H, v_head_dim]

        # --- 修正散布（Scatter）索引的计算 ---
        # 原始的 edge_dst 形状是 [E]
        # 为 weighted_v_flat 的每一行创建对应的边索引 e 和头索引 h
        e_indices = torch.arange(E * H, device=edge_dst.device) // H  # 形状 [E*H], 值像 [0, 0, 1, 1, ...]
        h_indices = torch.arange(E * H, device=edge_dst.device) % H  # 形状 [E*H], 值像 [0, 1, 0, 1, ...]
        # 将每个元素映射到目标输出行：目标节点索引 * H + 头索引
        scatter_index = edge_dst[e_indices] * H + h_indices # 形状 [E*H]
        # --- 修正结束 ---

        # 使用正确的索引进行 scatter_sum
        aggregated_v_flat = scatter_sum(weighted_v_flat.float(), scatter_index, dim=0, dim_size=N * H) # [N*H, v_head_dim]

        # Reshape 回 [N, H * v_head_dim] = [N, irreps_value.dim]
        aggregated_v = aggregated_v_flat.reshape(N, self.irreps_value.dim)

        # --- 6. Output Projection ---
        # Linear layer knows input/output irreps
        projected_output = self.output_projection(aggregated_v) # [N, irreps_node_output.dim]

        # --- 7. First Residual Connection ---
        if self.skip_projection_attn is not None:
            residual_input = self.skip_projection_attn(node_features)
        else:
            assert self.irreps_node_input == self.irreps_node_output, "Input/Output irreps mismatch without skip projection!"
            residual_input = node_features

        attn_block_output = residual_input + projected_output

        # --- 8. Feed-Forward Network ---
        ffn_output = self.ffn(attn_block_output)

        # --- 9. Second Residual Connection ---
        final_output = attn_block_output + ffn_output

        return final_output

# --- 主等变 GNN 编码器 (SE3InvariantGraphEncoder - Adjusted for Tensors) ---
class SE3InvariantGraphEncoder(nn.Module):
    def __init__(self, num_atom_types: int = 100, embedding_dim_scalar: int = 16, irreps_node_hidden: str = "64x0e + 16x1o + 8x2e", irreps_node_output: str = "128x0e", irreps_edge_attr: str = "0x0e", irreps_sh: str = "1x0e + 1x1o + 1x2e", max_radius: float = 5.0, num_basis_radial: int = 16, radial_mlp_hidden: list = [64, 64], num_interaction_layers: int = 3, num_attn_heads: int = 4, use_attention: bool = True, activation_gate: bool = True):
        super().__init__()
        self.irreps_node_input = o3.Irreps(f"{embedding_dim_scalar}x0e")
        self.node_embedding = nn.Embedding(num_atom_types, embedding_dim_scalar)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.max_radius = max_radius
        self.num_basis_radial = num_basis_radial
        self.use_attention = use_attention

        self.spherical_harmonics = SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')
        radial_input_dim = 1
        if self.irreps_edge_attr.dim > 0: radial_input_dim += self.irreps_edge_attr.dim
        self.radial_embedding = RadialMLP(radial_input_dim, radial_mlp_hidden, self.num_basis_radial)

        self.interaction_layers = nn.ModuleList()
        current_irreps = self.irreps_node_input

        for i in range(num_interaction_layers):
            if i == 0: in_irreps = self.irreps_node_input
            else: in_irreps = self.irreps_node_hidden

            if i < num_interaction_layers - 1: out_irreps = self.irreps_node_hidden
            else: out_irreps = self.irreps_node_output

            if use_attention:
                 block = SE3TransformerInteractionBlock(
                     irreps_node_input=str(in_irreps), irreps_node_output=str(out_irreps),
                     irreps_edge_attr=str(self.irreps_edge_attr), irreps_sh=str(self.irreps_sh),
                     num_basis_radial=self.num_basis_radial, radial_mlp_hidden=radial_mlp_hidden,
                     num_attn_heads=num_attn_heads, activation_gate=activation_gate,
                 )
            else:
                 block = TFNInteractionBlock(
                     irreps_node_input=str(in_irreps), irreps_node_output=str(out_irreps),
                     irreps_edge_attr=str(self.irreps_edge_attr), irreps_sh=str(self.irreps_sh),
                     num_basis_radial=self.num_basis_radial, radial_mlp_hidden=radial_mlp_hidden,
                     activation_gate=activation_gate
                 )
            self.interaction_layers.append(block)

        # --- Final Invariant Projection Logic (Updated for Tensor output) ---
        self.final_invariant_projection_layers = None # Reset
        self.final_scalar_dim = 0
        last_layer_output_irreps = o3.Irreps(out_irreps) # Get Irreps of last layer output
        if not all(ir.l == 0 for _, ir in last_layer_output_irreps):
             print(f"Warning: Final node output irreps ({last_layer_output_irreps}) contain non-scalar features. Adding Norm projection.")
             self.final_invariant_projection_layers = nn.ModuleDict()
             scalar_feature_sources = []

             original_scalars = last_layer_output_irreps.filter(l=0)
             if original_scalars.dim > 0:
                  scalar_feature_sources.append({'type': 'scalar', 'irreps': str(original_scalars)})

             non_scalars = last_layer_output_irreps.filter(l>0)
             for mul, ir in non_scalars:
                  norm_key = f'norm_{mul}x{ir}'
                  # Norm layer needs input irreps, output is mul x 0e
                  self.final_invariant_projection_layers[norm_key] = Norm(o3.Irreps([(mul, ir)])) # Input is specific irrep
                  scalar_feature_sources.append({'type': 'norm', 'key': norm_key, 'irreps': str(o3.Irreps([(mul, ir)]))}) # Store input irreps

             self.final_scalar_dim = sum(o3.Irreps(src['irreps']).dim if src['type'] == 'scalar' else o3.Irreps(src['irreps']).num_irreps for src in scalar_feature_sources)
             print(f"Pooling will use scalars derived from norms, total scalar dim: {self.final_scalar_dim}")
        else:
             self.final_scalar_dim = last_layer_output_irreps.dim

    def forward(self, data: Data) -> torch.Tensor: # Return Tensor
        pos = data.pos.float(); node_indices = data.x.long(); edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        node_scalars = self.node_embedding(node_indices)
        # Initial features are just scalars [N, C], need to match input irreps "Cx0e"
        h = node_scalars # Start with the tensor

        edge_src, edge_dst = edge_index
        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_vec_normalized = torch.zeros_like(edge_vec)
        valid_edge = edge_dist.squeeze() > 1e-6
        edge_vec_normalized[valid_edge] = edge_vec[valid_edge] / edge_dist[valid_edge]
        # SH layer expects tensor input
        edge_sh = self.spherical_harmonics(edge_vec_normalized).to(h.device)

        if edge_attr is not None and self.irreps_edge_attr.dim > 0:
            radial_input = torch.cat([edge_dist, edge_attr.float()], dim=-1)
        else:
            radial_input = edge_dist
        edge_radial_emb = self.radial_embedding(radial_input).to(h.device)

        # Pass tensors through layers
        current_irreps = self.irreps_node_input # Track irreps manually
        for i, interaction_block in enumerate(self.interaction_layers):
            h = interaction_block(h, edge_index, edge_sh, edge_radial_emb, edge_length=edge_dist, max_radius=self.max_radius)
            # Update current_irreps based on block's output
            current_irreps = interaction_block.irreps_node_output

        node_output_features_tensor = h # Tensor after last layer

        # --- Apply final invariant projection if needed ---
        if self.final_invariant_projection_layers is not None:
            scalar_features_list = []
            last_layer_output_irreps = current_irreps # Use tracked irreps

            # Process original scalars first
            original_scalars_irreps = last_layer_output_irreps.filter(l=0)
            if original_scalars_irreps.dim > 0:
                 scalar_part_tensor = o3.IrrepsArray(original_scalars_irreps, node_output_features_tensor).slice_by_irreps(original_scalars_irreps).array.squeeze(-1) # Extract scalar tensor data
                 scalar_features_list.append(scalar_part_tensor)

            # Process non-scalars using Norm layers
            non_scalars_irreps = last_layer_output_irreps.filter(l>0)
            for mul, ir in non_scalars_irreps:
                 norm_key = f'norm_{mul}x{ir}'
                 if norm_key in self.final_invariant_projection_layers:
                     norm_layer = self.final_invariant_projection_layers[norm_key]
                     # Apply norm layer to the corresponding slice of the tensor
                     feature_part_irreps = o3.Irreps([(mul, ir)])
                     feature_part_tensor = o3.IrrepsArray(last_layer_output_irreps, node_output_features_tensor).slice_by_irreps(feature_part_irreps) # Slice returns IrrepsArray
                     norms_tensor = norm_layer(feature_part_tensor).array # Norm returns IrrepsArray, get tensor
                     scalar_features_list.append(norms_tensor)

            if not scalar_features_list:
                 return torch.zeros(node_output_features_tensor.shape[0], 0, device=node_output_features_tensor.device)
            node_invariant_features = torch.cat(scalar_features_list, dim=-1) # [N, final_scalar_dim]
        else:
            # If output is already scalar
            node_invariant_features = node_output_features_tensor.squeeze(-1) # Assume shape [N, dim, 1]

        return node_invariant_features # Return node-level features


# --- 测试代码 ---
if __name__ == '__main__':
    # ... (Test code remains the same) ...
    print("\n" + "="*30 + "\n--- Running matchecon.py Internal Tests ---" + "\n" + "="*30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_nodes = 15; batch_size = 3; num_atom_types = 10; embedding_dim = 16
    max_radius_test = 3.0; irreps_hidden = "32x0e + 8x1o" # Simplified
    irreps_output = "64x0e"; lmax_sh = 1
    irreps_sh_test = o3.Irreps.spherical_harmonics(lmax_sh); num_layers = 2; heads = 2

    print("\n--- Creating Mock Data ---")
    pos = torch.randn(N_nodes, 3, device=device) * max_radius_test / 2
    node_atom_types = torch.randint(0, num_atom_types, (N_nodes,), device=device)
    edge_index = radius_graph(pos, r=max_radius_test, loop=False)
    batch_idx = torch.zeros(N_nodes, dtype=torch.long, device=device)
    nodes_per_graph = N_nodes // batch_size
    for i in range(batch_size):
        start = i * nodes_per_graph
        end = (i + 1) * nodes_per_graph if i < batch_size - 1 else N_nodes
        batch_idx[start:end] = i

    mock_data = Data(x=node_atom_types, pos=pos, edge_index=edge_index, batch=batch_idx).to(device)
    print(f"Mock data created: {mock_data}")
    print(f"Number of edges: {mock_data.edge_index.shape[1]}")


    # --- Test TFN Mode ---
    print("\n" + "-"*20 + "\nTesting SE3InvariantGraphEncoder (TFN Mode)\n" + "-"*20)
    try:
        tfn_encoder = SE3InvariantGraphEncoder(
            num_atom_types=num_atom_types, embedding_dim_scalar=embedding_dim,
            irreps_node_hidden=irreps_hidden, irreps_node_output=irreps_output,
            irreps_edge_attr="0x0e", irreps_sh=str(irreps_sh_test),
            max_radius=max_radius_test, num_interaction_layers=num_layers,
            use_attention=False
        ).to(device)
        print("TFN Encoder instantiated.")
        tfn_output = tfn_encoder(mock_data)
        print(f"TFN Output shape: {tfn_output.shape}")
        expected_output_dim = o3.Irreps(irreps_output).dim
        assert tfn_output.shape == (N_nodes, expected_output_dim), f"Shape mismatch"
        assert not torch.isnan(tfn_output).any() and not torch.isinf(tfn_output).any(), "NaN/Inf detected"
        print("TFN Mode test passed basic checks.")

        print("Testing TFN Invariance...")
        out_original = tfn_encoder(mock_data)
        rot_matrix = o3.rand_matrix().to(device)
        mock_data_rotated = mock_data.clone()
        mock_data_rotated.pos = mock_data.pos @ rot_matrix.T
        out_rotated = tfn_encoder(mock_data_rotated)
        print(f"Comparing original output norm: {torch.norm(out_original).item():.4f}")
        print(f"Comparing rotated output norm: {torch.norm(out_rotated).item():.4f}")
        assert torch.allclose(out_original, out_rotated, atol=1e-4), "TFN output is NOT invariant!"
        print("TFN Invariance test passed!")

    except Exception as e:
        print(f"ERROR during TFN Mode test:")
        import traceback
        traceback.print_exc()


    # --- Test SE(3)T Mode ---
    print("\n" + "-"*20 + "\nTesting SE3InvariantGraphEncoder (Attention Mode)\n" + "-"*20)
    try:
        heads_test = 2
        se3t_hidden_irreps = "32x0e + 8x1o" # Ensure mul are even for H=2
        se3t_output_irreps = "64x0e"        # Output scalar

        se3t_encoder = SE3InvariantGraphEncoder(
            num_atom_types=num_atom_types, embedding_dim_scalar=embedding_dim,
            irreps_node_hidden=se3t_hidden_irreps,
            irreps_node_output=se3t_output_irreps,
            irreps_edge_attr="0x0e", irreps_sh=str(irreps_sh_test),
            max_radius=max_radius_test, num_interaction_layers=num_layers,
            num_attn_heads=heads_test,
            use_attention=True
        ).to(device)
        print("SE(3)T Encoder instantiated.")

        se3t_output = se3t_encoder(mock_data)
        print(f"SE(3)T Output shape: {se3t_output.shape}")
        expected_output_dim_se3t = o3.Irreps(se3t_output_irreps).dim
        assert se3t_output.shape == (N_nodes, expected_output_dim_se3t), f"Shape mismatch"
        assert not torch.isnan(se3t_output).any() and not torch.isinf(se3t_output).any(), "NaN/Inf detected"
        print("SE(3)T Mode test passed basic checks.")

        print("Testing SE(3)T Invariance...")
        out_original_se3t = se3t_encoder(mock_data)
        mock_data_rotated_se3t = mock_data.clone()
        mock_data_rotated_se3t.pos = mock_data.pos @ rot_matrix.T
        out_rotated_se3t = se3t_encoder(mock_data_rotated_se3t)
        print(f"Comparing original output norm: {torch.norm(out_original_se3t).item():.4f}")
        print(f"Comparing rotated output norm: {torch.norm(out_rotated_se3t).item():.4f}")
        assert torch.allclose(out_original_se3t, out_rotated_se3t, atol=1e-4), "SE(3)T output is NOT invariant!"
        print("SE(3)T Invariance test passed!")

    except Exception as e:
        print(f"ERROR during SE(3)T Mode test:")
        import traceback
        traceback.print_exc()

    print("\n" + "="*30 + "\n--- matchecon.py Internal Tests Finished ---" + "\n" + "="*30)