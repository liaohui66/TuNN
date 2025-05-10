# data_utils.py
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from pymatgen.core import Structure, Composition, Element
from monty.serialization import loadfn
from sklearn.model_selection import train_test_split
from typing import Union
import traceback 

# --- 1. 定义辅助类和函数 ---
class GaussianDistanceConverter:
    """将距离列表扩展为高斯基函数特征。"""
    def __init__(self, centers=None, width=None):
        if centers is None:
            centers = np.linspace(0, 5, 100)
        if width is None:
            width = 0.5
        self.centers = torch.tensor(centers, dtype=torch.float)
        self.width = width
        if self.width <= 0: raise ValueError("Gaussian width must be positive.")
        self.variance = self.width ** 2

    def convert(self, distances: torch.Tensor) -> torch.Tensor:
        """将距离张量转换为高斯特征张量。"""
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(- (diff ** 2) / self.variance)


class AtomInfoExtractor:
    """
    从 embedding 文件加载元素列表，并提供获取原子序数 Z 和 one-hot 编码的方法。
    """
    def __init__(self, embedding_path: str):
        print(f"Initializing AtomInfoExtractor with: {embedding_path}")
        self.embedding_path = Path(embedding_path)
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Atom embedding file not found at: {self.embedding_path}")

        try:
            with open(self.embedding_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            self.element_list = sorted(list(embedding_data.keys()))
            self.num_elements = len(self.element_list)
            if self.num_elements != 103:
                 print(f"Warning: Expected 103 elements based on model input, but found {self.num_elements} in {embedding_path}")
            self.element_to_index = {element: i for i, element in enumerate(self.element_list)}
            print(f"Found {self.num_elements} elements. Created one-hot mapping.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or parse elements from {self.embedding_path}: {e}")

    def get_atom_numbers(self, structure: Structure) -> torch.Tensor:
        atom_zs = []
        for site in structure.sites:
            if isinstance(site.specie, Composition):
                first_element = next(iter(site.species.elements))
                atom_zs.append(first_element.Z)
            else:
                atom_zs.append(site.specie.Z)
        return torch.tensor(atom_zs, dtype=torch.long)

    def get_one_hot_encoding(self, structure: Structure) -> torch.Tensor:
        one_hot_features = []
        for site in structure.sites:
            site_one_hot = torch.zeros(self.num_elements, dtype=torch.float32)
            total_occupancy = 0.0
            unknown_element_present = False
            for element, occupancy in site.species.items():
                symbol = str(element.symbol)
                if symbol in self.element_to_index:
                    index = self.element_to_index[symbol]
                    site_one_hot[index] = float(occupancy)
                    total_occupancy += float(occupancy)
                else:
                    print(f"Warning: Element '{symbol}' not in element list derived from embedding file. Cannot create one-hot.")
                    unknown_element_present = True
                    break

            if unknown_element_present:
                site_one_hot.zero_()
            elif total_occupancy > 0 and abs(total_occupancy - 1.0) > 1e-5:
                print(f"Warning: Site occupancy {total_occupancy:.3f} != 1 for {site.species}. Normalizing one-hot.")
                site_one_hot /= total_occupancy
            elif total_occupancy == 0:
                 print(f"Warning: Site {site.species} has zero known element occupancy.")

            one_hot_features.append(site_one_hot)

        if not one_hot_features:
             return torch.empty((0, self.num_elements), dtype=torch.float32)

        return torch.stack(one_hot_features)


def structure_to_pyg_graph(structure: Structure, atom_info_extractor: AtomInfoExtractor, cutoff: float,
                           gaussian_converter: GaussianDistanceConverter) -> Union[PyGData, None]:
    """
    将 Pymatgen Structure 对象转换为 PyTorch Geometric Data 对象。
    """
    try:
        if atom_info_extractor is None: raise ValueError("atom_info_extractor cannot be None.")
        if gaussian_converter is None: raise ValueError("gaussian_converter cannot be None.")

        N = len(structure)
        if N == 0:
            print(f"Warning: Structure {structure.formula} has zero atoms. Skipping.")
            return None

        node_zs = atom_info_extractor.get_atom_numbers(structure)
        node_one_hot = atom_info_extractor.get_one_hot_encoding(structure)

        if node_zs is None or node_zs.shape[0] == 0 or node_one_hot is None or node_one_hot.shape[0] == 0:
            print(f"Warning: Failed to get node features for structure {structure.formula}. Skipping.")
            return None

        atom_symbols = [str(site.specie) for site in structure.sites]

        neighbor_list = structure.get_neighbor_list(r=cutoff, numerical_tol=1e-8, exclude_self=True)
        center_indices, neighbor_indices, images, edge_distances = neighbor_list

        if len(center_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(gaussian_converter.centers)), dtype=torch.float32)
            edge_offset = torch.empty((0, 3), dtype=torch.long)
        else:
            edge_index = torch.tensor([center_indices, neighbor_indices], dtype=torch.long)
            edge_attr = gaussian_converter.convert(torch.tensor(edge_distances, dtype=torch.float))
            edge_offset = torch.tensor(images, dtype=torch.long)

        positions = torch.tensor(structure.cart_coords, dtype=torch.float32)

        comp_dict = Composition(structure.formula).get_el_amt_dict()
        total_atoms_in_formula = sum(comp_dict.values())
        element_weights = {Element(el).symbol: amt / total_atoms_in_formula
                           for el, amt in comp_dict.items()}

        atom_comp_weights = []
        for site in structure.sites:
            if isinstance(site.specie, Composition):
                first_element_symbol = next(iter(site.species.elements)).symbol
                weight = element_weights.get(first_element_symbol, 0.0)
            else:
                weight = element_weights.get(site.specie.symbol, 0.0)
            atom_comp_weights.append(weight)

        if len(atom_comp_weights) != N:
             print(f"Warning: Length mismatch after calculating atom weights ({len(atom_comp_weights)}) vs N ({N}).")
             atom_comp_weights.extend([0.0] * (N - len(atom_comp_weights)))

        comp_w_tensor = torch.tensor(atom_comp_weights, dtype=torch.float32).view(-1, 1)
        if comp_w_tensor.shape[0] != N:
             raise RuntimeError(f"Final comp_w shape {comp_w_tensor.shape} doesn't match number of nodes {N}")

        graph_data = PyGData(
            x=node_zs,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            atom_fea=node_one_hot,
            comp_w=comp_w_tensor,
            edge_offset=edge_offset,
            num_nodes=N,
            symbols=atom_symbols
        )
        return graph_data

    except Exception as e:
        print(f"ERROR during graph conversion for structure {structure.formula}: {e}")
        traceback.print_exc()
        return None

# --- PyTorch Dataset 类 ---
class SENPyTorchDataset(Dataset):

    def __init__(self, data_json_path: str, target_json_path: str, embedding_path: str,
                 cutoff: float = 5.0, gaussian_width: float = 0.5, n_gaussian_centers: int = 100,
                 max_gaussian_dist: float = 5.0,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 random_seed: int = 66,
                 phase: str = 'train',
                 # target_transform: str = 'none', # <-- 修改默认值
                 # target_norm: str = 'mean_std'
                 ):
        super().__init__()

        print(f"Initializing SENPyTorchDataset for phase: {phase}")
        assert phase in ['train', 'val', 'test'], "phase must be 'train', 'val', or 'test'"
        assert abs(sum(split_ratios) - 1.0) < 1e-6 and all(0 <= r <= 1 for r in split_ratios), "Split ratios invalid."
        if phase == 'test' and split_ratios[2] < 1e-6:
             print("Warning: phase='test' requested but test split ratio is zero.")

        self.cutoff = cutoff
        # self.target_transform_method = 'none' # 不再需要单独处理变换
        self.target_norm_method = 'global_max' # 内部强制使用全局最大值缩放
        self.phase = phase

        self.atom_info_extractor = AtomInfoExtractor(embedding_path=embedding_path)
        self.gaussian_converter = GaussianDistanceConverter(
            centers=np.linspace(0, max_gaussian_dist, n_gaussian_centers),
            width=gaussian_width
        )

        # --- 1. 加载原始数据 ---
        self.structures, self.targets_raw = self._load_and_merge_data(data_json_path, target_json_path)
        if not self.structures:
            raise ValueError("No valid structures loaded.")
        self.targets_raw = np.array(self.targets_raw).astype(float)

        # --- 2. 计算并应用全局 Max Scaling ---
        if self.targets_raw.size > 0:
            self.global_target_max = np.max(self.targets_raw) # 计算全局最大值
            if abs(self.global_target_max) < 1e-8:
                print("Warning: Global maximum target value is near zero. Setting max to 1.0 for scaling.")
                self.global_target_max = 1.0
            print(f"Applying Global Max Scaling using max={self.global_target_max:.4f}")
            # 直接修改 self.targets_raw，将其归一化
            self.targets_scaled = self.targets_raw / self.global_target_max
        else:
            print("Warning: No targets loaded, cannot compute global max.")
            self.global_target_max = 1.0 # 设置默认值以防出错
            self.targets_scaled = np.array([]) # 创建空数组

        # --- 3. 分割数据索引 ---
        self.split_indices = self._split_data(len(self.structures), split_ratios, random_seed)
        self.current_indices = self.split_indices[self.phase]

        print(f"Dataset initialized for phase '{self.phase}' with {len(self)} samples.")


    def _load_and_merge_data(self, struct_json_path, target_json_path):
        structures_map = {}
        targets_map = {}

        with open(struct_json_path, 'r', encoding='utf-8') as f:
            struct_raw_data = json.load(f)
        for entry in struct_raw_data:
            if isinstance(entry, dict) and 'material_id' in entry and 'cif' in entry:
                try:
                    struct = Structure.from_str(entry['cif'], fmt="cif")
                    structures_map[entry['material_id']] = struct
                except Exception as e:
                    print(f"Warning: Failed to parse CIF for ID {entry.get('material_id', 'N/A')}. Error: {e}")

        with open(target_json_path, 'r', encoding='utf-8') as f:
            target_raw_dict = json.load(f)
        for mat_id, target_val in target_raw_dict.items():
            try:
                targets_map[mat_id] = float(target_val)
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric target for ID {mat_id}. Value: {target_val}")

        final_structures = []
        final_targets = []
        for mat_id, struct in structures_map.items():
            if mat_id in targets_map:
                final_structures.append(struct)
                final_targets.append(targets_map[mat_id])

        print(f"Successfully merged {len(final_structures)} structure-target pairs.")
        return final_structures, np.array(final_targets)


    def _split_data(self, num_samples, ratios, seed):
        indices = np.arange(num_samples)
        train_ratio, val_ratio, test_ratio = ratios

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
             raise ValueError("Split ratios must sum to 1.")

        if abs(val_ratio + test_ratio) < 1e-6:
             train_indices = indices
             val_indices = np.array([], dtype=indices.dtype)
             test_indices = np.array([], dtype=indices.dtype)
        else:
             train_indices, remaining_indices = train_test_split(
                 indices, train_size=train_ratio, random_state=seed, shuffle=True)

             if abs(test_ratio) < 1e-6:
                 val_indices = remaining_indices
                 test_indices = np.array([], dtype=indices.dtype)
             elif abs(val_ratio) < 1e-6:
                 val_indices = np.array([], dtype=indices.dtype)
                 test_indices = remaining_indices
             else:
                 relative_val_ratio = val_ratio / (val_ratio + test_ratio)
                 val_indices, test_indices = train_test_split(
                     remaining_indices, train_size=relative_val_ratio, random_state=seed, shuffle=True)

        print(f"Data split ratios: Train={len(train_indices)/num_samples:.2f} ({len(train_indices)}), "
              f"Val={len(val_indices)/num_samples:.2f} ({len(val_indices)}), "
              f"Test={len(test_indices)/num_samples:.2f} ({len(test_indices)})")
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        actual_idx = self.current_indices[idx]
        structure = self.structures[actual_idx]
        # 获取【已经全局缩放过】的目标值
        target_scaled = self.targets_scaled[actual_idx]
        # 同时获取原始目标值，用于验证
        target_raw = self.targets_raw[actual_idx] / self.global_target_max # 从归一化值反算回原始 (有点绕，但保持了原始值)
                                                                          # 或者直接 self.targets_raw[actual_idx] 如果不修改原始数组
        target_raw_original = self.targets_raw[actual_idx] # 直接获取未修改的原始值


        graph_data = structure_to_pyg_graph(
            structure, self.atom_info_extractor, self.cutoff, self.gaussian_converter
        )
        if graph_data is None:
            # 返回一个包含必要字段的空 Data 对象，或者根据你的 DataLoader 处理方式返回 None
            # 这里返回一个带默认值的空 Data 对象
            return PyGData(x=torch.empty((0, self.atom_info_extractor.num_elements)), # 示例，确保 x 维度正确
                           edge_index=torch.empty((2, 0), dtype=torch.long),
                           edge_attr=torch.empty((0, self.gaussian_converter.centers.numel())),
                           pos=torch.empty((0, 3)),
                           y=torch.tensor([0.0], dtype=torch.float32), # 提供默认值
                           y_raw=torch.tensor([0.0], dtype=torch.float32), # 提供默认值
                           idx=torch.tensor([-1], dtype=torch.long)) # 标记为无效索引


        # --- [修改] data.y 现在是全局 Max Scaling 后的值 ---
        graph_data.y = torch.tensor([target_scaled], dtype=torch.float32)
        # --- [修改] data.y_raw 仍然是原始值 ---
        graph_data.y_raw = torch.tensor([target_raw_original], dtype=torch.float32) # 使用未修改的原始值
        graph_data.idx = torch.tensor([actual_idx], dtype=torch.long)

        return graph_data

    def denormalize(self, y_scaled_tensor):
        """
        将模型输出 (全局 Max Scaling 尺度) 反算回原始尺度。
        """
        if not isinstance(y_scaled_tensor, torch.Tensor):
            y_scaled_tensor = torch.tensor(y_scaled_tensor)

        if self.global_target_max is None:
             raise ValueError("Global max (global_target_max) not set for denormalization.")

        # 直接乘以全局最大值
        y_original = y_scaled_tensor * self.global_target_max

        return y_original


# --- 主程序示例 (需要更新参数) ---
if __name__ == '__main__':
    BASE_DIR = Path('.')
    DATA_DIR = BASE_DIR / 'data'
    DATA_JSON_PATH = DATA_DIR / 'data.json'
    TARGET_JSON_PATH = DATA_DIR / 'eij_data.json'
    EMBEDDING_PATH = DATA_DIR / 'matscholar-embedding.json'
    CUTOFF = 5.0
    GAUSSIAN_WIDTH = 0.5
    N_GAUSSIAN_CENTERS = 100
    MAX_GAUSSIAN_DIST = 5.0
    SPLIT_RATIOS = (0.8, 0.0, 0.2)
    RANDOM_SEED = 66
    # --- 修改: 设置默认 norm 和移除 transform ---
    TARGET_NORM_METHOD = 'max' # 或 'mean_std', 'min_max', 'none'
    TARGET_TRANSFORM_METHOD = 'none' # 明确设置为 none
    
    print("\n--- Testing Training Dataset (with Global Max Scaling) ---")
    train_dataset = None
    global_max_from_train = None # 用于传递给测试集
    try:
        train_dataset = SENPyTorchDataset(
            # ... (其他参数不变) ...
            phase='train',
            # target_norm='none' # 可以不传或传 'none'
        )
        print(f"Number of training samples: {len(train_dataset)}")
        global_max_from_train = train_dataset.global_target_max # 获取计算出的全局最大值
        print(f"Global Max Target used for scaling: {global_max_from_train:.4f}")

        if len(train_dataset) > 0:
            first_train_sample = train_dataset[0]
            # ... (打印和检查 first_train_sample, y, y_raw, denormalize) ...
            # 确保 denormalize(y) 接近 y_raw
            if hasattr(first_train_sample, 'y') and hasattr(first_train_sample, 'y_raw'):
                 denorm_y = train_dataset.denormalize(first_train_sample.y)
                 print(f"  Denormalized y: {denorm_y.item():.4f} vs y_raw: {first_train_sample.y_raw.item():.4f}")
                 assert np.isclose(denorm_y.item(), first_train_sample.y_raw.item()), "Denormalize check failed!"
                 print("  Denormalization check passed!")


    except Exception as e:
        print(f"\nERROR creating/testing training dataset: {e}")
        traceback.print_exc()

    print("\n--- Testing Test Dataset (with Global Max Scaling) ---")
    test_dataset = None
    if train_dataset and global_max_from_train is not None: # 确保训练集和全局最大值有效
        try:
            test_dataset = SENPyTorchDataset(
                 # ... (其他参数不变) ...
                 phase='test',
                 # target_norm='none'
             )
            # --- [修改] 不再需要 set_normalization_stats ---
            # test_dataset.set_normalization_stats(...)
            # --- [关键] 确保测试集知道用于反归一化的全局最大值 ---
            test_dataset.global_target_max = global_max_from_train
            print(f"Test dataset set to use Global Max Target: {test_dataset.global_target_max:.4f}")


            print(f"Number of test samples: {len(test_dataset)}")
            if len(test_dataset) > 0:
                 first_test_sample = test_dataset[0]
                 # ... (打印和检查 first_test_sample, y, y_raw, denormalize) ...
                 if hasattr(first_test_sample, 'y') and hasattr(first_test_sample, 'y_raw'):
                    denorm_y = test_dataset.denormalize(first_test_sample.y)
                    print(f"  Denormalized y: {denorm_y.item():.4f} vs y_raw: {first_test_sample.y_raw.item():.4f}")
                    assert np.isclose(denorm_y.item(), first_test_sample.y_raw.item()), "Denormalize check failed!"
                    print("  Denormalization check passed!")

        except Exception as e:
            print(f"\nERROR creating/testing test dataset: {e}")
            traceback.print_exc()
    else:
        print("Skipping test dataset test because training dataset failed or global max is None.")


    # --- 测试 DataLoader ---
    print("\n--- Testing DataLoader with PyG ---")
    # [--- BEGIN MODIFICATION ---]
    # --- 使用 train_dataset (如果它存在) ---
    if train_dataset and len(train_dataset) > 0: # 检查 train_dataset 是否成功创建且不为空
        try:
            from torch_geometric.loader import DataLoader as PyGDataLoader

            train_loader = PyGDataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
            print(f"Number of batches in train_loader: {len(train_loader)}")

            first_batch = next(iter(train_loader))
            print("\nFirst batch (PyG Batch object):")
            print(first_batch)
            print(f"  Number of graphs in batch: {first_batch.num_graphs}")
            print(f"  Batch vector shape: {first_batch.batch.shape}")
            print(f"  Node features shape: {first_batch.x.shape}")
            print(f"  Edge index shape: {first_batch.edge_index.shape}")
            print(f"  Edge attributes shape: {first_batch.edge_attr.shape}")
            # pos 可能不存在，添加检查
            if hasattr(first_batch, 'pos'): print(f"  Positions shape: {first_batch.pos.shape}")
            print(f"  Target (y) shape: {first_batch.y.shape}")

        except ImportError:
             print("\nPyTorch Geometric DataLoader not found. Skipping DataLoader test.")
        except Exception as e:
            print(f"\nERROR testing DataLoader: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping DataLoader test because training dataset was not created or is empty.")
