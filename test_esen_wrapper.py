"""
Simple test script for eSEN wrapper without HORM dependencies.
"""

import torch
from torch_geometric.data import Data

# Test import
print("Testing eSEN wrapper import...")
from nets.eSEN.esen_wrapper import ESENWrapper

print("✓ Import successful!")

# Create a simple test batch
print("\nCreating test batch...")
pos = torch.randn(5, 3, requires_grad=True)  # 5 atoms
z = torch.tensor([1, 6, 6, 8, 1], dtype=torch.long)  # H, C, C, O, H
batch_idx = torch.zeros(5, dtype=torch.long)  # Single molecule

test_batch = Data(
    pos=pos,
    z=z,
    batch=batch_idx,
)

print("✓ Test batch created!")

# Load model
print("\nLoading eSEN model...")
model = ESENWrapper(
    checkpoint_path='ckpt/esen_sm_direct_all.pt',
    device='cpu',  # Use CPU for testing
)
print("✓ Model loaded!")

# Test forward_autograd
print("\nTesting forward_autograd...")
energy, forces = model.forward_autograd(test_batch)

print(f"✓ Forward pass successful!")
print(f"  Energy shape: {energy.shape}")
print(f"  Forces shape: {forces.shape}")
print(f"  Energy has grad: {energy.requires_grad}")
print(f"  Forces has grad: {forces.requires_grad}")

print("\n✅ All tests passed!")
