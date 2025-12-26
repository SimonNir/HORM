"""
Zero-shot evaluation of eSEN on HORM dataset.

This script evaluates the pretrained eSEN model on HORM data without any fine-tuning.
It computes energy, force, and Hessian MAE metrics.
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from ff_lmdb import LmdbDataset
from nets.eSEN.esen_wrapper import ESENWrapper
from training_module import compute_extra_props


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    if forces is None:
        forces = -torch.autograd.grad(
            [energy.sum()], [coords], create_graph=True, retain_graph=True
        )[0]
    
    n_comp = forces.reshape(-1).shape[0]
    hess = []
    
    for f in forces.reshape(-1):
        hess_row = torch.autograd.grad(
            [-f], [coords], retain_graph=True, create_graph=False
        )[0]
        hess.append(hess_row)
    
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def hess2eigenvalues(hess):
    """Convert Hessian to eigenvalues."""
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


def evaluate_esen(checkpoint_path, lmdb_path, device='cuda'):
    """
    Evaluate eSEN model on HORM dataset.
    
    Args:
        checkpoint_path: Path to eSEN checkpoint (.pt file)
        lmdb_path: Path to HORM LMDB dataset
        device: Device to run on ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"eSEN Zero-shot Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {lmdb_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = ESENWrapper(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    model.eval()
    
    # Load dataset
    dataset = LmdbDataset(lmdb_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize metrics
    total_e_error = 0.0
    total_f_error = 0.0
    total_h_error = 0.0
    total_eigen_error = 0.0
    total_asymmetry_error = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = batch.to(device)
            batch.pos.requires_grad_(True)
            batch = compute_extra_props(batch)
            
            # Forward pass - use forward_autograd for smooth forces
            with torch.enable_grad():
                energy, forces = model.forward_autograd(batch)
            
            # Compute Hessian
            with torch.enable_grad():
                batch.pos.requires_grad_(True)
                energy_for_hess, forces_for_hess = model.forward_autograd(batch)
                hess = compute_hessian(batch.pos, energy_for_hess, forces_for_hess)
                eigenvalues = hess2eigenvalues(hess)
            
            # Compute errors
            e_error = torch.mean(torch.abs(energy.squeeze() - batch.ae))
            f_error = torch.mean(torch.abs(forces - batch.forces))
            
            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_error = torch.mean(torch.abs(hess - hessian_true))
            
            # Eigenvalue error
            eigen_true = hess2eigenvalues(hessian_true)
            eigen_error = torch.mean(torch.abs(eigenvalues - eigen_true))
            
            # Asymmetry error
            asymmetry_error = torch.mean(torch.abs(hess - hess.T))
            
            # Update totals
            total_e_error += e_error.item()
            total_f_error += f_error.item()
            total_h_error += h_error.item()
            total_eigen_error += eigen_error.item()
            total_asymmetry_error += asymmetry_error.item()
            n_samples += 1
            
            # Memory management
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Calculate average errors
    mae_e = total_e_error / n_samples
    mae_f = total_f_error / n_samples
    mae_h = total_h_error / n_samples
    mae_eigen = total_eigen_error / n_samples
    mae_asymmetry = total_asymmetry_error / n_samples
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Energy MAE:      {mae_e:.6f}")
    print(f"Forces MAE:      {mae_f:.6f}")
    print(f"Hessian MAE:     {mae_h:.6f}")
    print(f"Eigenvalue MAE:  {mae_eigen:.6f}")
    print(f"Asymmetry MAE:   {mae_asymmetry:.6f}")
    print(f"{'='*60}\n")
    
    return {
        'mae_e': mae_e,
        'mae_f': mae_f,
        'mae_h': mae_h,
        'mae_eigen': mae_eigen,
        'mae_asymmetry': mae_asymmetry,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate eSEN on HORM dataset')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='ckpt/esen_sm_conserving_all.pt',
        help='Path to eSEN checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/sample_100.lmdb',
        help='Path to HORM LMDB dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run evaluation
    results = evaluate_esen(
        checkpoint_path=args.checkpoint,
        lmdb_path=args.data,
        device=args.device,
    )
    
    # Save results
    checkpoint_name = Path(args.checkpoint).stem
    results_file = f'results_esen_{checkpoint_name}.txt'
    with open(results_file, 'w') as f:
        f.write(f"eSEN Zero-shot Evaluation Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data}\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
