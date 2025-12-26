"""
eSEN model wrapper for HORM framework - Direct model access for autograd Hessians.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from omegaconf import OmegaConf

try:
    from fairchem.core.models.base import HydraModel
    from fairchem.core.common.registry import registry
except ImportError:
    raise ImportError(
        "fairchem-core is required for eSEN. Install with: pip install fairchem-core"
    )


class ESENWrapper(nn.Module):
    """
    Wrapper for eSEN model that integrates with HORM's training framework.
    
    Loads model directly from checkpoint to enable full autograd support for Hessians.
    """
    
    def __init__(
        self,
        checkpoint_path: str = 'ckpt/esen_sm_direct_all.pt',
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\\n"
                f"Please download from HuggingFace: facebook/OMol25"
            )
        
        # Load checkpoint
        print(f"Loading eSEN checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model config and state dict
        model_config = ckpt.model_config
        state_dict = ckpt.model_state_dict
        
        # Convert config to dict if it's OmegaConf
        if hasattr(model_config, '_metadata'):
            # It's an OmegaConf object
            config_dict = OmegaConf.to_container(model_config, resolve=True)
        else:
            # It's already a dict
            config_dict = model_config
        
        # Remove _target_ key if present (Hydra uses this for instantiation)
        config_dict = dict(config_dict)  # Make a copy
        config_dict.pop('_target_', None)
        
        # Instantiate model from config
        self.model = HydraModel(**config_dict)
        
        # Load weights
        self.model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"eSEN model loaded successfully!")
        print(f"  Backbone: {config_dict.get('backbone', {}).get('model', 'N/A')}")
        print(f"  Heads: {list(config_dict.get('heads', {}).keys())}")
        
        # Detect if this is a conserving model (forces via autograd) or direct model
        heads = list(config_dict.get('heads', {}).keys())
        self.is_conserving = 'energyandforcehead' in heads
        self.is_direct = 'forces' in heads and 'energy' in heads
        
        if self.is_conserving:
            print(f"  Type: Conserving (forces via autograd)")
        elif self.is_direct:
            print(f"  Type: Direct (forces predicted directly)")

        
    def _prepare_batch(self, batch):
        """Convert HORM batch to fairchem format."""
        # Ensure we have the required fields
        if not hasattr(batch, 'z'):
            raise ValueError("Batch must have 'z' (atomic numbers) attribute")
        
        # Get number of graphs
        num_graphs = batch.batch.max().item() + 1
        natoms = torch.bincount(batch.batch)
        
        # Create cell for non-periodic molecules (large box)
        max_pos = batch.pos.abs().max()
        box_size = max_pos * 3 + 10.0
        cell = torch.eye(3, device=batch.pos.device) * box_size
        cell = cell.unsqueeze(0).repeat(num_graphs, 1, 1)
        
        # Compute edge_index using radius graph
        from torch_geometric.nn import radius_graph
        cutoff = 6.0  # eSEN cutoff
        edge_index = radius_graph(
            batch.pos,
            r=cutoff,
            batch=batch.batch,
            max_num_neighbors=30,  # eSEN max_neighbors
        )
        
        # Compute edge distances and offsets
        num_edges = edge_index.shape[1]
        cell_offsets = torch.zeros(num_edges, 3, device=batch.pos.device)
        nedges = torch.bincount(batch.batch[edge_index[0]], minlength=num_graphs)
        
        # Create batch dict for fairchem model
        fairchem_batch = {
            'pos': batch.pos,
            'atomic_numbers': batch.z.long(),
            'natoms': natoms,
            'batch': batch.batch,
            'cell': cell,
            'charge': torch.zeros(num_graphs, dtype=torch.long, device=batch.pos.device),
            'spin': torch.ones(num_graphs, dtype=torch.long, device=batch.pos.device),
            'dataset': torch.zeros(num_graphs, dtype=torch.long, device=batch.pos.device),  # Default dataset ID
            'edge_index': edge_index,
            'cell_offsets': cell_offsets,
            'nedges': nedges,
        }
        
        return fairchem_batch
    
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with HORM's training framework.
        
        Args:
            batch: PyG Data object from HORM with pos, z, batch, etc.
            
        Returns:
            Tuple of (energy, forces)
        """
        # Prepare batch
        fairchem_batch = self._prepare_batch(batch)
        
        # Forward through model
        output = self.model(fairchem_batch)
        
        # Extract energy and forces from output dict
        energy = output.get('energy', None)
        forces = output.get('forces', None)
        
        if energy is None:
            raise ValueError("Model did not return energy")
        
        # Ensure correct shapes
        if isinstance(energy, dict):
            energy = energy.get('energy', energy)
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        
        if forces is None:
            # If model doesn't provide forces, return zeros
            forces = torch.zeros_like(batch.pos)
        elif isinstance(forces, dict):
            forces = forces.get('forces', forces)
        
        return energy, forces
    
    def forward_autograd(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that computes forces via autograd with create_graph=True.
        This enables Hessian computation via second-order autograd.
        
        For conserving models: Calls backbone + energy head directly to avoid
        internal force computation, then computes forces via autograd.
        
        For direct models: Gets energy from full forward, then recomputes forces
        via autograd for consistency.
        
        Args:
            batch: PyG Data object from HORM
            
        Returns:
            Tuple of (energy, forces) where forces = -dE/dpos with gradients
        """
        # Enable gradients on positions
        batch.pos.requires_grad_(True)
        
        # Prepare batch
        fairchem_batch = self._prepare_batch(batch)
        
        # Forward through model with gradients enabled
        with torch.enable_grad():
            if self.is_conserving:
                # For conserving models, bypass the full forward to avoid internal autograd
                # Call backbone to get embeddings
                emb = self.model.backbone(fairchem_batch)
                
                # Get the energy head (it's called 'energyandforcehead' but we only use energy part)
                energy_head = self.model.output_heads['energyandforcehead']
                
                # Compute per-node energies using the energy block
                node_energy = energy_head.energy_block(
                    emb['node_embedding'].narrow(1, 0, 1).squeeze(1)
                ).view(-1, 1, 1)
                
                # Reduce to per-graph energy
                energy = torch.zeros(
                    len(fairchem_batch['natoms']),
                    device=fairchem_batch['pos'].device,
                    dtype=node_energy.dtype,
                )
                energy.index_add_(0, fairchem_batch['batch'], node_energy.view(-1))
            else:
                # For direct models, use full forward pass to get energy
                output = self.model(fairchem_batch)
                energy = output.get('energy', None)
                
                if energy is None:
                    raise ValueError("Model did not return energy")
                
                # Handle nested dict
                if isinstance(energy, dict):
                    energy = energy.get('energy', energy)
                
                if energy.dim() == 0:
                    energy = energy.unsqueeze(0)
        
        # Compute forces via autograd with create_graph=True for Hessian support
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=batch.pos,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return energy, forces
