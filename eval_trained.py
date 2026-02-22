"""
Post-hoc evaluation of trained eSEN checkpoints (from train_esen_comparison.py).

Evaluates E, F, and H MAE for all three trained variants on the validation set,
using full autograd Hessians for all models regardless of training objective.

This enables a fair comparison:
  - E-only model: autograd forces and hessians (though not trained on them)
  - E+F model:    autograd forces and hessians (trained on E+F)
  - E+F+H model:  autograd forces and hessians (trained on E+F+H)

Usage:
    # Evaluate a single checkpoint
    python eval_trained.py --checkpoint checkpoint/run/best.ckpt --data data/ts1x-val.lmdb

    # Evaluate all variants in a run directory
    python eval_trained.py --run_dir checkpoint/horm-esen-comparison/ --data data/ts1x-val.lmdb
"""

import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from ff_lmdb import LmdbDataset
from training_module import PotentialModule, compute_extra_props


def compute_full_hessian(forces, pos, num_rows=None):
    """
    Compute Hessian rows via autograd.

    Args:
        forces: (N, 3) force tensor with grad_fn
        pos: (N, 3) positions, requires_grad=True
        num_rows: if set, only compute this many randomly sampled rows (faster)

    Returns:
        hessian: (K, N*3) where K = num_rows or N*3
        row_indices: which rows of the full (N*3, N*3) hessian were computed
    """
    n_dof = forces.numel()  # N * 3
    forces_flat = forces.reshape(-1)

    if num_rows is None or num_rows >= n_dof:
        row_indices = list(range(n_dof))
    else:
        row_indices = torch.randperm(n_dof)[:num_rows].tolist()

    hess_rows = []
    for idx in row_indices:
        row = torch.autograd.grad(
            outputs=-forces_flat[idx],  # -dE/dpos -> dF/dpos = -d2E/dpos2
            inputs=pos,
            retain_graph=True,
            create_graph=False,
        )[0]
        hess_rows.append(row.reshape(-1))

    hessian = torch.stack(hess_rows, dim=0)  # (K, N*3)
    return hessian, row_indices


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, data_path, device, num_hessian_rows=None, batch_size=1):
    """
    Load a trained PL checkpoint and evaluate E, F, H MAE on a dataset.

    Args:
        checkpoint_path: Path to Lightning .ckpt file
        data_path: Path to LMDB validation set
        device: 'cuda' or 'cpu'
        num_hessian_rows: rows of hessian to sample per molecule (None = full)
        batch_size: evaluation batch size (1 recommended for hessian eval)

    Returns:
        dict with mae_e, mae_f, mae_h
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {Path(checkpoint_path).name}")
    print(f"Data:       {data_path}")
    print(f"{'='*60}")

    # Load PL checkpoint - this reconstructs the full PotentialModule
    pm = PotentialModule.load_from_checkpoint(checkpoint_path, map_location=device)
    pm = pm.to(device)
    pm.eval()

    dataset = LmdbDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_e, total_f, total_h = 0.0, 0.0, 0.0
    n = 0

    for batch in tqdm(loader, desc='Eval'):
        batch = batch.to(device)
        batch.pos.requires_grad_(True)

        with torch.enable_grad():
            batch = compute_extra_props(batch, pos_require_grad=True)
            energy, forces = pm.potential.forward_autograd(batch)

            # Compute hessian rows for this batch
            hess_pred, row_indices = compute_full_hessian(
                forces, batch.pos, num_rows=num_hessian_rows
            )

        # Ground truth
        ae_true = batch.ae.to(device)
        forces_true = batch.forces.to(device)

        # Reconstruct true hessian rows
        n_atoms = batch.pos.shape[0]
        hess_true_full = batch.hessian.reshape(n_atoms * 3, n_atoms * 3).to(device)
        hess_true_rows = hess_true_full[row_indices]

        e_err = torch.mean(torch.abs(energy.squeeze() - ae_true)).item()
        f_err = torch.mean(torch.abs(forces - forces_true)).item()
        h_err = torch.mean(torch.abs(hess_pred - hess_true_rows)).item()

        total_e += e_err
        total_f += f_err
        total_h += h_err
        n += 1

    results = {
        'mae_e': total_e / n,
        'mae_f': total_f / n,
        'mae_h': total_h / n,
        'n_samples': n,
        'checkpoint': str(checkpoint_path),
        'data': str(data_path),
    }

    print(f"Energy MAE:  {results['mae_e']:.6f}")
    print(f"Forces MAE:  {results['mae_f']:.6f}")
    print(f"Hessian MAE: {results['mae_h']:.6f}  (rows sampled: {num_hessian_rows or 'all'})")
    return results


def find_best_checkpoints(run_dir):
    """Find best checkpoint for each variant (e, ef, efh) under run_dir."""
    run_dir = Path(run_dir)
    checkpoints = {}
    for ckpt in sorted(run_dir.rglob('*.ckpt')):
        for variant in ['efh', 'ef', 'e']:  # order matters - check efh before ef
            if f'esen-{variant}-' in ckpt.name:
                # Pick checkpoint with lowest val-totloss (encoded in filename)
                if variant not in checkpoints:
                    checkpoints[variant] = ckpt
                else:
                    # Keep the one with lower loss in filename
                    try:
                        cur_loss = float(str(checkpoints[variant]).split('val-totloss=')[1].split('-')[0])
                        new_loss = float(str(ckpt).split('val-totloss=')[1].split('-')[0])
                        if new_loss < cur_loss:
                            checkpoints[variant] = ckpt
                    except (IndexError, ValueError):
                        pass
                break
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained eSEN checkpoints on E/F/H')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=str, help='Path to single .ckpt file')
    group.add_argument('--run_dir', type=str, help='Directory containing multiple checkpoints')

    parser.add_argument('--data', type=str, required=True, help='Path to validation LMDB')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_hessian_rows', type=int, default=None,
                        help='Hessian rows per molecule (None = full; use e.g. 10 for speed)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Eval batch size (1 recommended for hessian)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    all_results = {}

    if args.checkpoint:
        results = evaluate_checkpoint(
            args.checkpoint, args.data, args.device,
            args.num_hessian_rows, args.batch_size
        )
        all_results[Path(args.checkpoint).stem] = results
    else:
        # Evaluate all found checkpoints
        checkpoints = find_best_checkpoints(args.run_dir)
        if not checkpoints:
            print(f"No .ckpt files found under {args.run_dir}")
            return

        print(f"Found checkpoints: {list(checkpoints.keys())}")
        for variant, ckpt_path in sorted(checkpoints.items()):
            results = evaluate_checkpoint(
                ckpt_path, args.data, args.device,
                args.num_hessian_rows, args.batch_size
            )
            all_results[variant] = results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<12} {'MAE_E':>12} {'MAE_F':>12} {'MAE_H':>12}")
    print('-' * 50)
    for name, r in all_results.items():
        print(f"{name:<12} {r['mae_e']:>12.6f} {r['mae_f']:>12.6f} {r['mae_h']:>12.6f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
