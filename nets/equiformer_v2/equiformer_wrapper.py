"""
Wrappers for EquiformerV2_OC20 compatible with the HORM training framework.

Two variants:
  EquiformerV2Wrapper            — direct force head (forces predicted by a dedicated head)
  EquiformerV2ConservativeWrapper — conservative forces (-dE/dpos via autograd)

Both expose the same interface as ESENWrapperScratch:
  forward(batch)          -> (energy, forces)
  forward_autograd(batch) -> (energy, forces) with create_graph=True for Hessian training

EquiformerV2_OC20 already accepts HORM-style PyG batches directly (uses data.z, data.batch,
data.natoms, data.ae) so no format conversion is needed.

Note: EquiformerV2_OC20.forward() always computes grad_hess_ij internally (legacy code that
is computed but not returned). This is wasteful but harmless for correctness.
"""

import torch
import torch.nn as nn
from typing import Tuple

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20


# Small model config suitable for comparison study on sample_100.lmdb.
# Matches roughly the parameter count of a small eSEN.
_SMALL_CONFIG = dict(
    use_pbc=False,
    regress_forces=True,
    otf_graph=True,
    max_neighbors=20,
    max_radius=6.0,
    max_num_elements=90,
    num_layers=4,
    sphere_channels=64,
    attn_hidden_channels=32,
    num_heads=4,
    attn_alpha_channels=32,
    attn_value_channels=16,
    ffn_hidden_channels=64,
    norm_type='rms_norm_sh',
    lmax_list=[2],
    mmax_list=[2],
    grid_resolution=18,
    num_sphere_samples=128,
    edge_channels=16,
    use_atom_edge_embedding=True,
    share_atom_edge_embedding=False,
    use_m_share_rad=False,
    distance_function='gaussian',
    num_distance_basis=512,
    attn_activation='scaled_silu',
    use_s2_act_attn=False,
    use_attn_renorm=True,
    ffn_activation='scaled_silu',
    use_gate_act=False,
    use_grid_mlp=True,
    use_sep_s2_act=True,
    alpha_drop=0.0,
    drop_path_rate=0.0,
    proj_drop=0.0,
    weight_init='uniform',
)


class EquiformerV2Wrapper(nn.Module):
    """
    EquiformerV2 with direct force prediction.

    Forces are predicted by a dedicated SO(2) equivariant force head.
    forward_autograd() re-computes forces via -dE/dpos for Hessian training.
    """

    def __init__(self, **model_kwargs):
        super().__init__()
        cfg = {**_SMALL_CONFIG, **model_kwargs}
        cfg['regress_forces'] = True
        self.model = EquiformerV2_OC20(**cfg)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"EquiformerV2 (direct forces): {total/1e6:.2f}M params")

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Direct force prediction."""
        energy, forces = self.model(batch)
        return energy, forces

    def forward_autograd(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autograd forces with create_graph=True for Hessian training.
        Runs the full forward (energy from energy_block), then recomputes
        forces as -dE/dpos so the Hessian autograd graph is intact.
        """
        batch.pos.requires_grad_(True)
        with torch.enable_grad():
            energy, _ = self.model(batch)
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=batch.pos,
            create_graph=True,
            retain_graph=True,
        )[0]
        return energy, forces


class EquiformerV2ConservativeWrapper(nn.Module):
    """
    EquiformerV2 with conservative (autograd) force prediction.

    Forces are always computed as -dE/dpos — no direct force head is used.
    This is the conservative variant: energy is the fundamental quantity,
    forces and Hessians are derived via automatic differentiation.
    """

    def __init__(self, **model_kwargs):
        super().__init__()
        cfg = {**_SMALL_CONFIG, **model_kwargs}
        # regress_forces=True keeps the force_block in the model graph
        # (required by EquiformerV2_OC20 constructor), but we ignore its output.
        cfg['regress_forces'] = True
        self.model = EquiformerV2_OC20(**cfg)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"EquiformerV2Conservative (autograd forces): {total/1e6:.2f}M params")

    def _get_energy(self, batch) -> torch.Tensor:
        energy, _ = self.model(batch)
        return energy

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conservative forward: forces = -dE/dpos (no graph kept)."""
        batch.pos.requires_grad_(True)
        with torch.enable_grad():
            energy = self._get_energy(batch)
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=batch.pos,
            create_graph=False,
            retain_graph=False,
        )[0]
        return energy, forces

    def forward_autograd(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conservative forward with create_graph=True for Hessian training."""
        batch.pos.requires_grad_(True)
        with torch.enable_grad():
            energy = self._get_energy(batch)
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=batch.pos,
            create_graph=True,
            retain_graph=True,
        )[0]
        return energy, forces
