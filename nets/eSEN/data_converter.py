"""
Data conversion utilities between HORM and fairchem formats.
"""

import torch
from typing import Dict, Any


def horm_to_fairchem(batch) -> Dict[str, Any]:
    """
    Convert HORM PyG Data batch to fairchem AtomicData format.
    
    Args:
        batch: PyG Data object from HORM LMDB with attributes:
            - pos: (N, 3) atomic positions
            - z: (N,) atomic numbers
            - batch: (N,) batch indices
            - ae: (B,) atomization energies
            - forces: (N, 3) atomic forces
            - hessian: (N*3, N*3) or flattened hessian
            
    Returns:
        Dictionary compatible with fairchem's AtomicData format
    """
    # Basic atomic data
    data = {
        'pos': batch.pos.clone(),
        'atomic_numbers': batch.z.long(),
        'natoms': batch.batch.bincount(),
        'batch': batch.batch,
    }
    
    # Add cell for non-periodic systems (molecules)
    # eSEN expects a cell even for molecules
    if not hasattr(batch, 'cell') or batch.cell is None:
        # Create a large box for non-periodic molecules
        max_pos = batch.pos.abs().max()
        box_size = max_pos * 3 + 10.0  # Add padding
        data['cell'] = torch.eye(3, device=batch.pos.device) * box_size
        data['cell'] = data['cell'].unsqueeze(0).repeat(data['natoms'].shape[0], 1, 1)
    else:
        data['cell'] = batch.cell
    
    # Add energy if available
    if hasattr(batch, 'ae'):
        data['energy'] = batch.ae
    elif hasattr(batch, 'energy'):
        data['energy'] = batch.energy
        
    # Add forces if available
    if hasattr(batch, 'forces'):
        data['forces'] = batch.forces
        
    return data


def fairchem_to_horm(fairchem_output: Dict[str, torch.Tensor], original_batch) -> tuple:
    """
    Convert fairchem model output back to HORM format.
    
    Args:
        fairchem_output: Dictionary with 'energy' and optionally 'forces'
        original_batch: Original HORM batch for reference
        
    Returns:
        Tuple of (energy, forces) in HORM format
    """
    energy = fairchem_output.get('energy', None)
    forces = fairchem_output.get('forces', None)
    
    # Ensure correct shapes
    if energy is not None:
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        elif energy.dim() == 1:
            pass  # Already correct shape (batch_size,)
        else:
            energy = energy.squeeze()
            
    if forces is not None:
        # Forces should be (N, 3)
        if forces.dim() != 2 or forces.shape[1] != 3:
            raise ValueError(f"Unexpected forces shape: {forces.shape}")
    
    return energy, forces
