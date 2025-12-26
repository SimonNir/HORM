"""
Quick test to verify eSEN conserving model works with Hessian computation.
"""

import torch
from torch_geometric.data import Data
from nets.eSEN.esen_wrapper import ESENWrapper

print("Testing eSEN conserving model with Hessian computation...")

# Create a simple test batch
pos = torch.randn(5, 3, requires_grad=True)
z = torch.tensor([1, 6, 6, 8, 1], dtype=torch.long)
batch_idx = torch.zeros(5, dtype=torch.long)

test_batch = Data(
    pos=pos,
    z=z,
    batch=batch_idx,
)

# Load conserving model
model = ESENWrapper(
    checkpoint_path='ckpt/esen_sm_conserving_all.pt',
    device='cpu',
)

print("\n1. Testing forward_autograd...")
energy, forces = model.forward_autograd(test_batch)
print(f"✓ Energy shape: {energy.shape}, requires_grad: {energy.requires_grad}")
print(f"✓ Forces shape: {forces.shape}, requires_grad: {forces.requires_grad}")

print("\n2. Testing Hessian computation (second-order autograd)...")
# Compute a single Hessian element: d²E/dpos[0,0]²
hess_element = torch.autograd.grad(
    outputs=forces[0, 0],
    inputs=test_batch.pos,
    retain_graph=True,
)[0]
print(f"✓ Hessian element shape: {hess_element.shape}")
print(f"✓ Hessian computation successful!")

print("\n3. Testing HVP (Hessian-vector product) like HORM does...")
# Sample a random vector
v = torch.randn_like(forces[0])
# Compute Jacobian-vector product
jvp = torch.autograd.grad(
    outputs=forces[0],
    inputs=test_batch.pos,
    grad_outputs=v,
    retain_graph=True,
)[0]
print(f"✓ HVP shape: {jvp.shape}")
print(f"✓ HVP computation successful!")

print("\n✅ All tests passed! eSEN conserving model works with Hessians!")
print("\nThis confirms:")
print("  - Energy has gradients")
print("  - Forces computed via autograd have gradients")
print("  - Second-order derivatives (Hessians) work")
print("  - HVP computation works (what HORM uses for training)")
