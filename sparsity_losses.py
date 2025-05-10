# sparsity_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Assuming helper functions are available in modules.geo_torch
try:
    from geometry.geo_torch import normalize_torch, safe_ce_torch
except ImportError:
    # Provide simple implementations if geo_torch is not available/complete yet
    print("Warning: Could not import from modules.geo_torch. Using placeholder implementations for normalize_torch and safe_ce_torch.")
    def normalize_torch(tensor: torch.Tensor, axis: int, eps: float = 1e-8) -> torch.Tensor:
        return tensor / (torch.sum(tensor, dim=axis, keepdim=True) + eps)

    def safe_ce_torch(labels: torch.Tensor, probs: torch.Tensor, axis: int = -1, eps: float = 1e-16) -> torch.Tensor:
        safe_probs = torch.log(torch.clamp(probs, min=eps))
        # Note: tf.reduce_mean is often applied *outside* safe_ce in TF examples,
        # but the original TF code seems to imply mean is taken here.
        # We return the sum and let the caller take the mean if needed,
        # matching the TF _capsule_entropy structure more closely where mean is taken later.
        # Let's revert to taking the mean here for simplicity based on tf.nn.l2_loss behavior analysis.
        return torch.mean(-torch.sum(labels * safe_probs, dim=axis))


def _capsule_entropy_pytorch(caps_presence_prob: torch.Tensor, k: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes entropy in capsule activations (PyTorch version).
    Corresponds to TensorFlow's _capsule_entropy.

    Args:
        caps_presence_prob (Tensor): Capsule presence probabilities [batch_size, n_caps].
        k (float): Scaling factor for the target distribution in cross-entropy.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (within_example_loss, between_example_loss) - scalar tensors.
    """
    if caps_presence_prob.numel() == 0: # Handle empty batch
        return torch.tensor(0.0, device=caps_presence_prob.device), torch.tensor(0.0, device=caps_presence_prob.device)

    # Within-example sparsity: Entropy of prob distribution over capsules for each example
    within_example_probs = normalize_torch(caps_presence_prob, axis=1) # Normalize across capsules
    # CE(p, p*k) -> Encourages peaky distributions if k > 1? Or just entropy if k=1.
    # safe_ce_torch takes the mean over the batch dimension implicitly if axis=-1 (or axis=1 here)
    within_example_loss = safe_ce_torch(within_example_probs, within_example_probs * k, axis=1)

    # Between-example sparsity: Entropy of average capsule activation probability across the batch
    between_example_probs = torch.mean(caps_presence_prob, dim=0) # Average activation per capsule [n_caps]
    between_example_probs = normalize_torch(between_example_probs, axis=0) # Normalize across capsules
    # safe_ce_torch takes the mean, but input is 1D, so axis=0 gives scalar loss.
    between_example_loss = safe_ce_torch(between_example_probs, between_example_probs * k, axis=0)

    return within_example_loss, between_example_loss


def _neg_capsule_kl_pytorch(caps_presence_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes KL divergence from uniform distribution (PyTorch version).
    Corresponds to TensorFlow's _neg_capsule_kl.
    This is equivalent to negative entropy (plus const), achieved by setting k=num_caps in _capsule_entropy.

    Args:
        caps_presence_prob (Tensor): Capsule presence probabilities [batch_size, n_caps].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (within_example_loss, between_example_loss) - scalar tensors.
    """
    if caps_presence_prob.numel() == 0: # Handle empty batch
        return torch.tensor(0.0, device=caps_presence_prob.device), torch.tensor(0.0, device=caps_presence_prob.device)

    num_caps = caps_presence_prob.shape[1]
    if num_caps == 0: # Handle case with zero capsules
         return torch.tensor(0.0, device=caps_presence_prob.device), torch.tensor(0.0, device=caps_presence_prob.device)

    # KL(p || uniform) = log(k) - H(p). We calculate H(p) essentially.
    return _capsule_entropy_pytorch(caps_presence_prob, k=float(num_caps))


def _caps_pres_l2_pytorch(caps_presence_prob: torch.Tensor,
                          num_classes: Optional[float] = None, # TF default uses 10? Check config
                          within_example_constant: float = 0.0,
                          # between_example_constant is implicitly calculated
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes L2 penalty on capsule activations (PyTorch version).
    Corresponds to TensorFlow's _caps_pres_l2.

    Args:
        caps_presence_prob (Tensor): Capsule presence probabilities [batch_size, n_caps].
        num_classes (Optional[float]): Number of classes (used for default constants).
        within_example_constant (float): Target sum for within-example probs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (within_example_loss, neg_between_example_loss) - scalar tensors.
                                         Note the negative sign for between_example.
    """
    if caps_presence_prob.numel() == 0: # Handle empty batch
        return torch.tensor(0.0, device=caps_presence_prob.device), torch.tensor(0.0, device=caps_presence_prob.device)

    batch_size_f = float(caps_presence_prob.shape[0])
    num_caps_f = float(caps_presence_prob.shape[1])

    if num_caps_f == 0: # Handle case with zero capsules
        return torch.tensor(0.0, device=caps_presence_prob.device), torch.tensor(0.0, device=caps_presence_prob.device)

    # Calculate default constants if needed
    if num_classes is None:
        print("Warning: num_classes not provided for L2 sparsity. Defaulting to num_caps.")
        num_classes = num_caps_f # Or use a default like 10 if that's standard?

    if within_example_constant == 0.0 and num_classes > 0:
        within_example_constant = num_caps_f / num_classes

    # TF code calculates between_example_constant = batch_size_f / num_classes inside the loss term implicitly.

    # Within-example L2 loss: || sum_k(p_ik) - C_w ||^2 / B
    sum_within = torch.sum(caps_presence_prob, dim=1) # [batch_size]
    within_example_loss = torch.sum((sum_within - within_example_constant)**2) / batch_size_f

    # Between-example L2 loss: || sum_i(p_ik) - C_b ||^2 / K
    # C_b = B / num_classes
    sum_between = torch.sum(caps_presence_prob, dim=0) # [n_caps]
    between_example_constant = batch_size_f / num_classes
    between_example_loss = torch.sum((sum_between - between_example_constant)**2) / num_caps_f

    # Negate between_example_loss to match TF return format for subtraction later
    return within_example_loss, -between_example_loss


def sparsity_loss_pytorch(loss_type: str, caps_presence_prob: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes capsule sparsity loss according to the specified type (PyTorch dispatcher).

    Args:
        loss_type (str): Type of sparsity loss: 'entropy', 'kl', 'l2'.
        caps_presence_prob (Tensor): Capsule presence probabilities [batch_size, n_caps].
        **kwargs: Additional arguments passed to the specific loss function
                  (e.g., k for entropy, num_classes for l2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (within_example_loss, between_example_loss or neg_between_l2)
                                         Scalar tensors.
    """
    loss_type = loss_type.lower()

    if loss_type == 'entropy':
        # k defaults to 1.0 if not provided
        return _capsule_entropy_pytorch(caps_presence_prob, k=kwargs.get('k', 1.0))

    elif loss_type == 'kl':
        # KL divergence from uniform
        return _neg_capsule_kl_pytorch(caps_presence_prob)

    elif loss_type == 'l2':
        # num_classes defaults to None (will use num_caps)
        # within_example_constant defaults to 0.0 (will use num_caps/num_classes)
        return _caps_pres_l2_pytorch(caps_presence_prob,
                                     num_classes=kwargs.get('num_classes', None),
                                     within_example_constant=kwargs.get('within_example_constant', 0.0))

    else:
        raise ValueError(f'Invalid sparsity loss type: "{loss_type}"')


# --- Test Code ---
if __name__ == '__main__':
    print("--- Testing Sparsity Losses ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock data
    batch_size = 4
    n_caps = 10
    # Create probabilities (not necessarily summing to 1, simulating unnormalized output)
    mock_probs = torch.rand(batch_size, n_caps, device=device) * 0.8 + 0.1 # Values between 0.1 and 0.9

    print(f"\nMock Probs Shape: {mock_probs.shape}")

    # --- Test Entropy ---
    print("\nTesting loss_type='entropy' (k=1)...")
    try:
        within_ent, between_ent = sparsity_loss_pytorch('entropy', mock_probs, k=1.0)
        print(f"  Within-Example Entropy Loss: {within_ent.item():.4f}")
        print(f"  Between-Example Entropy Loss: {between_ent.item():.4f}")
        assert within_ent.dim() == 0 and between_ent.dim() == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # --- Test KL ---
    print("\nTesting loss_type='kl'...")
    try:
        within_kl, between_kl = sparsity_loss_pytorch('kl', mock_probs)
        print(f"  Within-Example KL Loss: {within_kl.item():.4f}")
        print(f"  Between-Example KL Loss: {between_kl.item():.4f}")
        assert within_kl.dim() == 0 and between_kl.dim() == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()


    # --- Test L2 ---
    print("\nTesting loss_type='l2' (num_classes=5)...")
    try:
        within_l2, neg_between_l2 = sparsity_loss_pytorch('l2', mock_probs, num_classes=5.0)
        print(f"  Within-Example L2 Loss: {within_l2.item():.4f}")
        print(f"  Neg Between-Example L2 Loss: {neg_between_l2.item():.4f}") # Remember it's negated
        assert within_l2.dim() == 0 and neg_between_l2.dim() == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n--- Testing with Empty Input ---")
    empty_probs = torch.empty((0, n_caps), device=device)
    try:
        within_empty, between_empty = sparsity_loss_pytorch('l2', empty_probs)
        print(f"  L2 Loss (empty): within={within_empty.item()}, between={between_empty.item()}")
        assert within_empty.item() == 0.0 and between_empty.item() == 0.0
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()


    print("\n--- Sparsity Loss Tests Finished ---")