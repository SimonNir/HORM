"""
eSEN model wrapper with support for training from scratch (random initialization).

This wrapper can either:
1. Load pretrained weights from checkpoint (transfer learning)
2. Initialize from scratch using the same architecture (random weights)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional
from omegaconf import OmegaConf

try:
    from fairchem.core.models.base import HydraModel
    from fairchem.core.common.registry import registry
except ImportError:
    raise ImportError(
        "fairchem-core is required for eSEN. Install with: pip install fairchem-core"
    )


class ESENWrapperScratch(nn.Module):
    """
    eSEN wrapper with support for training from scratch.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint (for config extraction)
        device: Device to run on
        load_weights: If True, load pretrained weights. If False, use random init.
        config_dict: Optional - provide config directly instead of loading from checkpoint
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = 'ckpt/esen_sm_conserving_all.pt',
        device: str = 'cuda',
        load_weights: bool = True,
        config_dict: Optional[dict] = None,
    ):
        super().__init__()
        
        self.device = device
        self.load_weights = load_weights
        
        # Get configuration
        if config_dict is not None:
            # Use provided config
            print("Using provided configuration...")
            self.config_dict = config_dict
            state_dict = None
        elif checkpoint_path is not None:
            # Load from checkpoint
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Please download from HuggingFace: facebook/OMol25"
                )
            
            print(f"Loading eSEN configuration from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Get model config
            model_config = ckpt.model_config
            
            # Convert config to dict if it's OmegaConf
            if hasattr(model_config, '_metadata'):
                self.config_dict = OmegaConf.to_container(model_config, resolve=True)
            else:
                self.config_dict = model_config
            
            # Get state dict if we're loading weights
            state_dict = ckpt.model_state_dict if load_weights else None
        else:
            raise ValueError("Must provide either checkpoint_path or config_dict")
        
        # Clean up config
        self.config_dict = dict(self.config_dict)
        self.config_dict.pop('_target_', None)
        
        # Instantiate model from config
        print(f"Instantiating eSEN model...")
        print(f"  Backbone: {self.config_dict.get('backbone', {}).get('model', 'N/A')}")
        print(f"  Heads: {list(self.config_dict.get('heads', {}).keys())}")
        
        self.model = HydraModel(**self.config_dict)
        
        # Load weights or use random initialization
        if load_weights and state_dict is not None:
            print(f"  Loading pretrained weights...")
            self.model.load_state_dict(state_dict)
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"  ✓ Loaded {total_params:,} parameters ({total_params/1e6:.2f}M)")
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  ✓ Initialized from scratch with {total_params:,} parameters ({total_params/1e6:.2f}M)")
            print(f"  ⚠ Using random initialization (training from scratch)")
        
        # Move to device
        self.model = self.model.to(device)
        
        # Detect model type
        heads = list(self.config_dict.get('heads', {}).keys())
        self.is_conserving = 'energyandforcehead' in heads
        self.is_direct = 'forces' in heads and 'energy' in heads
        
        if self.is_conserving:
            print(f"  Type: Conserving (forces via autograd)")
        elif self.is_direct:
            print(f"  Type: Direct (forces predicted directly)")
    
    def _prepare_batch(self, batch):
        """Convert HORM batch to fairchem format."""
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
            max_num_neighbors=30,
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
            'dataset': torch.zeros(num_graphs, dtype=torch.long, device=batch.pos.device),
            'edge_index': edge_index,
            'cell_offsets': cell_offsets,
            'nedges': nedges,
        }
        
        return fairchem_batch
    
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with HORM's training framework.
        
        Args:
            batch: PyG Data object from HORM
            
        Returns:
            Tuple of (energy, forces)
        """
        fairchem_batch = self._prepare_batch(batch)
        output = self.model(fairchem_batch)
        
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
            forces = torch.zeros_like(batch.pos)
        elif isinstance(forces, dict):
            forces = forces.get('forces', forces)
        
        return energy, forces
    
    def forward_autograd(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that computes forces via autograd with create_graph=True.
        This enables Hessian computation via second-order autograd.
        
        Args:
            batch: PyG Data object from HORM
            
        Returns:
            Tuple of (energy, forces) where forces = -dE/dpos with gradients
        """
        batch.pos.requires_grad_(True)
        fairchem_batch = self._prepare_batch(batch)
        
        with torch.enable_grad():
            if self.is_conserving:
                # For conserving models, bypass full forward
                emb = self.model.backbone(fairchem_batch)
                energy_head = self.model.output_heads['energyandforcehead']
                
                node_energy = energy_head.energy_block(
                    emb['node_embedding'].narrow(1, 0, 1).squeeze(1)
                ).view(-1, 1, 1)
                
                energy = torch.zeros(
                    len(fairchem_batch['natoms']),
                    device=fairchem_batch['pos'].device,
                    dtype=node_energy.dtype,
                )
                energy.index_add_(0, fairchem_batch['batch'], node_energy.view(-1))
            else:
                # For direct models, use full forward
                output = self.model(fairchem_batch)
                energy = output.get('energy', None)
                
                if energy is None:
                    raise ValueError("Model did not return energy")
                
                if isinstance(energy, dict):
                    energy = energy.get('energy', energy)
                
                if energy.dim() == 0:
                    energy = energy.unsqueeze(0)
        
        # Compute forces via autograd
        forces = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=batch.pos,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return energy, forces


# Convenience function to create from-scratch model
def create_esen_from_scratch(
    checkpoint_path: str = 'ckpt/esen_sm_conserving_all.pt',
    device: str = 'cuda'
) -> ESENWrapperScratch:
    """
    Create eSEN model with random initialization (training from scratch).
    
    Uses the architecture from the checkpoint but doesn't load the weights.
    
    Args:
        checkpoint_path: Path to checkpoint (used only for architecture config)
        device: Device to run on
        
    Returns:
        ESENWrapperScratch with random weights
    """
    return ESENWrapperScratch(
        checkpoint_path=checkpoint_path,
        device=device,
        load_weights=False  # Key parameter for from-scratch training
    )


# Convenience function to load pretrained model
def create_esen_pretrained(
    checkpoint_path: str = 'ckpt/esen_sm_conserving_all.pt',
    device: str = 'cuda'
) -> ESENWrapperScratch:
    """
    Create eSEN model with pretrained weights (transfer learning).
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        device: Device to run on
        
    Returns:
        ESENWrapperScratch with pretrained weights
    """
    return ESENWrapperScratch(
        checkpoint_path=checkpoint_path,
        device=device,
        load_weights=True  # Load pretrained weights
    )
