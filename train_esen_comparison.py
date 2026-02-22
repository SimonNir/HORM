"""
Unified training script for ESEN on HORM dataset.

This script trains ESEN from scratch with three different training strategies:
1. Energy-only (E)
2. Energy + Forces (E+F)
3. Energy + Forces + Hessian (E+F+H)

Each training run logs metrics for E, F, and H losses over time to enable comparison.

Usage:
    # Train all three variants sequentially
    python train_esen_comparison.py --mode all --data data/sample_100.lmdb
    
    # Train specific variant
    python train_esen_comparison.py --mode ef --data data/sample_100.lmdb
    
    # For cluster submission, use the accompanying SLURM script
"""

import argparse
from uuid import uuid4
import torch
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from training_module import PotentialModule


torch.set_float32_matmul_precision('high')


def train_variant(
    variant: str,
    data_path: str,
    val_data_path: str,
    checkpoint_path: str,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    project: str,
    use_wandb: bool = True,
    devices: int = 1,
    from_scratch: bool = False,
    resume_from: str = None,
):
    """
    Train a single variant (E, E+F, or E+F+H).
    
    Args:
        variant: One of 'e', 'ef', 'efh'
        data_path: Path to LMDB dataset
        checkpoint_path: Path to pretrained ESEN checkpoint (or config source if from_scratch=True)
        val_data_path: Path to validation LMDB dataset (may differ from data_path)
        batch_size: Batch size
        lr: Learning rate
        max_epochs: Maximum epochs
        patience: Early stopping patience
        project: WandB project name
        use_wandb: Whether to use WandB logging
        devices: Number of GPU devices
        from_scratch: If True, initialize with random weights. If False, load pretrained weights.
        resume_from: Path to a last.ckpt to resume from (restores weights, optimizer, epoch, etc.)
    """
    
    # Configuration based on variant
    if variant == 'e':
        use_hessian = False
        force_weight = 0.0
        energy_weight = 4.0
        hessian_weight = 0.0
        num_hessian_rows = 0
        description = "Energy-only"
        tags = ["energy-only", "E"]
    elif variant == 'ef':
        use_hessian = False
        force_weight = 100.0
        energy_weight = 4.0
        hessian_weight = 0.0
        num_hessian_rows = 0
        description = "Energy + Forces"
        tags = ["energy-force", "E+F"]
    elif variant == 'efh':
        use_hessian = True
        force_weight = 100.0
        energy_weight = 4.0
        hessian_weight = 4.0
        num_hessian_rows = 1  # 1 for conserving (autograd), 2 for direct
        description = "Energy + Forces + Hessian"
        tags = ["energy-force-hessian", "E+F+H"]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Add suffix to run name based on training mode
    training_mode = "scratch" if from_scratch else "pretrained"
    run_name = f"eSEN-{variant}-{training_mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"\n{'='*80}")
    print(f"Training ESEN: {description}")
    print(f"{'='*80}")
    print(f"Variant: {variant.upper()}")
    print(f"Training mode: {'FROM SCRATCH (random init)' if from_scratch else 'FINE-TUNING (pretrained)'}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"Data: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Loss weights - E: {energy_weight}, F: {force_weight}, H: {hessian_weight}")
    print(f"{'='*80}\n")
    
    # Model configuration
    model_config = dict(
        name="eSEN",
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_targets=1,
        output_dim=1,
        readout="sum",
        use_pbc=False,
        direct_forces=False,  # eSEN uses autograd for forces
        pos_require_grad=True,
        compute_forces=True,
        compute_stress=False,
        from_scratch=from_scratch,  # Key parameter: train from scratch or fine-tune
    )
    
    # Optimizer configuration
    optimizer_config = dict(
        lr=lr,
        betas=[0.9, 0.999],
        weight_decay=1e-5,
        amsgrad=True,
    )
    
    # Training configuration
    training_config = dict(
        trn_path=data_path,
        val_path=val_data_path,
        bz=batch_size,
        num_workers=8,
        clip_grad=True,
        gradient_clip_val=0.1,
        ema=False,
        use_hessian=use_hessian,
        hessian_weight=hessian_weight,
        num_hessian_rows=num_hessian_rows,
        force_weight=force_weight,
        energy_weight=energy_weight,
        lr_schedule_type="step",
        lr_schedule_config=dict(
            gamma=0.9,
            step_size=100,
        ),
    )
    
    # Initialize model (training from scratch by loading pretrained checkpoint)
    pm = PotentialModule(model_config, optimizer_config, training_config)
    
    # Setup logging
    loggers = []
    
    # CSV logger (always enabled for offline analysis)
    csv_logger = CSVLogger(
        save_dir=f"logs/{project}",
        name=run_name,
    )
    loggers.append(csv_logger)
    
    # WandB logger (optional)
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=project,
                log_model=False,
                name=run_name,
                tags=tags,
            )
            loggers.append(wandb_logger)
            ckpt_path = f"checkpoint/{project}/{wandb_logger.experiment.name}"
        except Exception as e:
            print(f"Warning: WandB logging failed ({e}). Continuing with CSV only.")
            ckpt_path = f"checkpoint/{project}/{run_name}"
    else:
        ckpt_path = f"checkpoint/{project}/{run_name}"
    
    # Create checkpoint directory
    os.makedirs(ckpt_path, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val-totloss",
        dirpath=ckpt_path,
        filename=f"esen-{variant}-{{epoch:03d}}-{{val-totloss:.4f}}-{{val-MAE_E:.4f}}-{{val-MAE_F:.4f}}",
        save_top_k=3,
        save_last=True,   # always keeps last.ckpt for easy restart after timeout
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val-totloss",
        patience=patience,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        TQDMProgressBar(),
        lr_monitor,
    ]
    
    # Trainer
    trainer = Trainer(
        devices=devices,
        num_nodes=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        max_epochs=max_epochs,
        callbacks=callbacks,
        default_root_dir=ckpt_path,
        logger=loggers,
        gradient_clip_val=0.1,
        accumulate_grad_batches=1,
    )
    
    # Train (resume_from restores weights, optimizer, LR scheduler, epoch, and callback state)
    trainer.fit(pm, ckpt_path=resume_from)
    
    print(f"\n{'='*80}")
    print(f"Training completed: {description}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Logs saved to: {ckpt_path}")
    print(f"{'='*80}\n")
    
    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train ESEN on HORM with different supervision levels")
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['e', 'ef', 'efh', 'all'],
        help='Training mode: e (energy-only), ef (energy+force), efh (energy+force+hessian), all (run all three)'
    )
    
    # Data and checkpoint
    parser.add_argument('--data', type=str, default='data/sample_100.lmdb', help='Path to training LMDB dataset')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation LMDB dataset (defaults to --data if not set)')
    parser.add_argument('--checkpoint', type=str, default='ckpt/esen_sm_conserving_all.pt', help='Pretrained ESEN checkpoint')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (adjust for E+F+H: 8-16)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    # Logging and system
    parser.add_argument('--project', type=str, default='horm-esen-comparison', help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--devices', type=int, default=1, help='Number of GPU devices')
    
    # Training mode
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch (random init) instead of fine-tuning pretrained weights')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='Path to last.ckpt to resume from (restores weights, optimizer, epoch, callbacks)')
    
    args = parser.parse_args()
    
    # Resolve val_data
    if args.val_data is None:
        args.val_data = args.data

    # Verify data exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Val data file not found: {args.val_data}")

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    
    print(f"\n{'#'*80}")
    print(f"# ESEN Training on HORM Dataset - Comparison Study")
    print(f"#")
    print(f"# Training mode: {'FROM SCRATCH (random init)' if args.from_scratch else 'FINE-TUNING (pretrained)'}")
    print(f"# This script will train ESEN with different supervision levels")
    print(f"# to compare learning dynamics.")
    print(f"#")
    print(f"# Train data: {args.data}")
    print(f"# Val data:   {args.val_data}")
    print(f"# Checkpoint: {args.checkpoint}")
    print(f"# Project: {args.project}")
    print(f"# WandB: {'Enabled' if not args.no_wandb else 'Disabled'}")
    print(f"{'#'*80}\n")
    
    # Train variants based on mode
    results = {}
    
    if args.mode == 'all':
        variants = ['e', 'ef', 'efh']
        # Adjust batch size for E+F+H (more memory intensive)
        batch_sizes = {
            'e': args.batch_size,
            'ef': args.batch_size,
            'efh': max(8, args.batch_size // 2)  # Reduce batch size for Hessian
        }
    else:
        variants = [args.mode]
        batch_sizes = {args.mode: args.batch_size}
    
    for variant in variants:
        best_ckpt = train_variant(
            variant=variant,
            data_path=args.data,
            val_data_path=args.val_data,
            checkpoint_path=args.checkpoint,
            batch_size=batch_sizes[variant],
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            project=args.project,
            use_wandb=not args.no_wandb,
            devices=args.devices,
            from_scratch=args.from_scratch,
            resume_from=args.resume,
        )
        results[variant] = best_ckpt
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# Training Complete - Summary")
    print(f"{'#'*80}")
    for variant, ckpt in results.items():
        print(f"  {variant.upper()}: {ckpt}")
    print(f"\n# Next steps:")
    print(f"  1. Compare training curves in WandB or CSV logs")
    print(f"  2. Evaluate best checkpoints on test set")
    print(f"  3. Analyze E, F, H metrics across training regimes")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
