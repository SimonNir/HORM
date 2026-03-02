"""
Wrapper for ESCaIP (Efficiently Scaled Attention Interatomic Potential) compatible
with the HORM training framework.

Uses EScAIPBackbone + EScAIPGradientEnergyForceStressHead (conservative forces via autograd).
Forces are always computed as -dE/dpos, making this a conservative model.

Exposes the same interface as ESENWrapperScratch:
  forward(batch)          -> (energy, forces)
  forward_autograd(batch) -> (energy, forces) with create_graph=True for Hessian training

HORM batches are converted to the format ESCaIP expects (atomic_numbers, pos, batch,
num_graphs) using a lightweight PyG Data wrapper.
"""

import torch
import torch.nn as nn
from typing import Tuple
from torch_geometric.data import Data


try:
    from fairchem.core.models.escaip.EScAIP import (
        EScAIPBackbone,
        EScAIPGradientEnergyForceStressHead,
    )
except ImportError:
    from nets.escaip.EScAIP import (
        EScAIPBackbone,
        EScAIPGradientEnergyForceStressHead,
    )


# Small ESCaIP config for comparison study on sample_100.lmdb.
# freequency_list must sum to hidden_size // atten_num_heads.
# hidden_size=128, atten_num_heads=4 -> head_dim=32 -> freequency_list sums to 32.
_SMALL_CONFIG = dict(
    # GlobalConfigs
    regress_forces=True,
    direct_forces=False,       # conservative: forces via autograd
    hidden_size=128,
    num_layers=4,
    activation='gelu',
    regress_stress=False,
    use_compile=False,         # disable torch.compile for robustness
    use_padding=False,         # disable padding for variable-size molecules
    use_fp16_backbone=False,
    dataset_list=[],
    # MolecularGraphConfigs
    use_pbc=False,
    max_num_elements=90,
    max_atoms=100,             # max atoms per molecule in the dataset
    max_batch_size=64,
    max_radius=6.0,
    knn_k=20,
    knn_soft=False,
    knn_sigmoid_scale=1.0,
    knn_lse_scale=1.0,
    knn_use_low_mem=False,
    knn_pad_size=0,
    distance_function='gaussian',
    use_envelope=True,
    # GraphNeuralNetworksConfigs
    atten_name='math',         # math attention required for gradient forces
    atten_num_heads=4,
    atom_embedding_size=128,
    node_direction_embedding_size=64,
    node_direction_expansion_size=10,
    edge_distance_expansion_size=600,
    edge_distance_embedding_size=512,
    readout_hidden_layer_multiplier=2,
    output_hidden_layer_multiplier=2,
    ffn_hidden_layer_multiplier=2,
    use_angle_embedding='none',
    use_graph_attention=False,
    use_message_gate=False,
    use_global_readout=False,
    use_frequency_embedding=True,
    freequency_list=[8, 4, 4, 8, 8],  # sums to 32 = 128 // 4
    energy_reduce='sum',
    # RegularizationConfigs
    normalization='rmsnorm',
    mlp_dropout=0.0,
    atten_dropout=0.0,
    stochastic_depth_prob=0.0,
    node_ffn_dropout=0.0,
    edge_ffn_dropout=0.0,
    scalar_output_dropout=0.0,
    vector_output_dropout=0.0,
)


class ESCaIPWrapper(nn.Module):
    """
    ESCaIP wrapper with conservative (autograd) force prediction.

    Uses EScAIPBackbone + EScAIPGradientEnergyForceStressHead.
    Forces are always computed as -dE/dpos.
    """

    def __init__(self, **model_kwargs):
        super().__init__()
        cfg = {**_SMALL_CONFIG, **model_kwargs}
        # Force conservative settings
        cfg['direct_forces'] = False
        cfg['regress_forces'] = True
        cfg['atten_name'] = 'math'   # required for gradient-based forces
        cfg['use_compile'] = False   # compile breaks autograd graph

        self.backbone = EScAIPBackbone(**cfg)
        self.head = EScAIPGradientEnergyForceStressHead(self.backbone)

        total = sum(p.numel() for p in self.parameters())
        print(f"ESCaIP (conservative): {total/1e6:.2f}M params")

    def _prepare_batch(self, batch) -> Data:
        """Convert HORM PyG batch to a Data object ESCaIP can consume."""
        if not hasattr(batch, 'z'):
            raise ValueError("Batch must have 'z' (atomic numbers). "
                             "Call compute_extra_props() first.")
        num_graphs = int(batch.batch.max().item()) + 1
        data = Data(
            pos=batch.pos,
            atomic_numbers=batch.z.long(),
            batch=batch.batch,
            num_graphs=num_graphs,
        )
        return data

    def _run(self, batch, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core forward: backbone -> head -> (energy, forces).
        Forces are computed via autograd inside the head; create_graph controls
        whether the second-order graph is kept (needed for Hessian).
        """
        data = self._prepare_batch(batch)
        data.pos.requires_grad_(True)

        with torch.enable_grad():
            emb = self.backbone(data)

            # Temporarily set training mode so the head uses create_graph=True
            # when we need it (the head uses create_graph=self.training internally).
            was_training = self.head.training
            if create_graph:
                self.head.train()

            result = self.head(data, emb)

            if not was_training:
                self.head.eval()

        energy = result.get('energy', None)
        forces = result.get('forces', None)

        if energy is None:
            raise RuntimeError("ESCaIP head did not return 'energy'")
        if forces is None:
            raise RuntimeError("ESCaIP head did not return 'forces' — "
                               "check that regress_forces=True and direct_forces=False")

        if energy.dim() == 0:
            energy = energy.unsqueeze(0)

        return energy, forces

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward (no Hessian graph kept)."""
        return self._run(batch, create_graph=False)

    def forward_autograd(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with create_graph=True for Hessian training."""
        return self._run(batch, create_graph=True)
