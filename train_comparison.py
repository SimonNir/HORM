"""
Model comparison training script for HORM.

Trains four models from scratch with two supervision levels each (EF, EFH):
  - eSEN                    (conserving, autograd forces)
  - EquiformerV2Direct      (direct force head)
  - EquiformerV2Conservative (conserving, autograd forces)
  - ESCaIP                  (conserving, autograd forces)

Designed for quick comparison on sample_100.lmdb.

Usage:
    # Train a single combination
    python train_comparison.py --model esen --mode ef --data data/sample_100.lmdb

    # Train all 8 combinations sequentially
    python train_comparison.py --model all --mode all --data data/sample_100.lmdb

    # Cluster submission
    sbatch submit_comparison.sh
"""

import argparse
import os
import torch
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from training_module import PotentialModule

torch.set_float32_matmul_precision('high')

MODELS = ['esen', 'equiformerv2', 'equiformerv2_conservative', 'escaip']
MODES  = ['ef', 'efh']

# Map CLI model name -> training_module model name
MODEL_NAME_MAP = {
    'esen':                     'eSEN',
    'equiformerv2':             'EquiformerV2Direct',
    'equiformerv2_conservative': 'EquiformerV2Conservative',
    'escaip':                   'ESCaIP',
}

# Per-model hyperparameters matching HORM paper conventions (notes_design_decisions.md §2, §5):
#   - Conserving/autograd models: LR=1e-4, NHR=1
#   - Direct-force EquiformerV2:  LR=3e-4, NHR=2
_MODEL_LR = {
    'esen':                      1e-4,
    'equiformerv2':              3e-4,
    'equiformerv2_conservative': 1e-4,
    'escaip':                    1e-4,
}
_MODEL_NHR = {
    'esen':                      1,
    'equiformerv2':              2,
    'equiformerv2_conservative': 1,
    'escaip':                    1,
}


def train_one(
    model_key: str,
    mode: str,
    data_path: str,
    val_data_path: str,
    esen_checkpoint: str,
    batch_size: int,
    lr: float,          # None → use per-model HORM convention from _MODEL_LR
    max_epochs: int,
    patience: int,
    project: str,
    devices: int,
    output_dir: str,
    use_wandb: bool = True,
    resume_from: str = None,
):
    """Train one (model, mode) combination."""

    model_name = MODEL_NAME_MAP[model_key]

    # Per-model LR: use HORM paper convention unless explicitly overridden
    effective_lr = lr if lr is not None else _MODEL_LR[model_key]

    # ── Supervision settings ────────────────────────────────────────────────
    if mode == 'ef':
        use_hessian    = False
        force_weight   = 100.0
        energy_weight  = 4.0
        hessian_weight = 0.0
        num_hessian_rows = 0
        mode_label = "E+F"
    elif mode == 'efh':
        use_hessian    = True
        force_weight   = 100.0
        energy_weight  = 4.0
        hessian_weight = 4.0
        # HORM paper: NHR=1 for conserving/autograd, NHR=2 for direct-force
        num_hessian_rows = _MODEL_NHR[model_key]
        mode_label = "E+F+H"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    run_name = f"{model_key}-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tags = [model_key, mode_label, "comparison"]

    print(f"\n{'='*80}")
    print(f"Training {model_name} | {mode_label} | from scratch")
    print(f"  Run:   {run_name}")
    print(f"  LR:    {effective_lr}  NHR: {num_hessian_rows}")
    print(f"  Data:  {data_path}")
    print(f"  Val:   {val_data_path}")
    print(f"{'='*80}\n")

    # ── Model config ─────────────────────────────────────────────────────────
    model_config = dict(name=model_name)
    if model_key == 'esen':
        model_config.update(
            checkpoint_path=esen_checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            from_scratch=True,
        )
    # EquiformerV2 and ESCaIP use default small configs defined in their wrappers.
    # Pass nothing extra — wrappers handle defaults.

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_config = dict(
        lr=effective_lr,
        betas=[0.9, 0.999],
        weight_decay=1e-5,
        amsgrad=True,
    )

    # ── Training config ───────────────────────────────────────────────────────
    training_config = dict(
        trn_path=data_path,
        val_path=val_data_path,
        bz=batch_size,
        num_workers=4,
        clip_grad=True,
        gradient_clip_val=0.1,
        ema=False,
        use_hessian=use_hessian,
        hessian_weight=hessian_weight,
        num_hessian_rows=num_hessian_rows,
        force_weight=force_weight,
        energy_weight=energy_weight,
        lr_schedule_type="step",
        lr_schedule_config=dict(gamma=0.9, step_size=50),
    )

    pm = PotentialModule(model_config, optimizer_config, training_config)

    # ── Logging ───────────────────────────────────────────────────────────────
    loggers = []
    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, "logs", project),
        name=run_name,
    )
    loggers.append(csv_logger)

    ckpt_path = os.path.join(output_dir, "checkpoint", project, run_name)

    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=project,
                log_model=False,
                name=run_name,
                tags=tags,
                save_dir=output_dir,
            )
            loggers.append(wandb_logger)
            ckpt_path = os.path.join(
                output_dir, "checkpoint", project, wandb_logger.experiment.name
            )
        except Exception as e:
            print(f"Warning: WandB logging failed ({e}). Continuing with CSV only.")

    os.makedirs(ckpt_path, exist_ok=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        monitor="val-totloss",
        dirpath=ckpt_path,
        filename=f"{model_key}-{mode}-{{epoch:03d}}-{{val-totloss:.4f}}",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val-totloss",
        patience=patience,
        mode="min",
    )
    callbacks = [checkpoint_cb, early_stop_cb, TQDMProgressBar(), LearningRateMonitor()]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        devices=devices,
        num_nodes=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if devices > 1 else "auto",
        max_epochs=max_epochs,
        callbacks=callbacks,
        default_root_dir=ckpt_path,
        logger=loggers,
        gradient_clip_val=0.1,
        precision="bf16-mixed",
    )

    trainer.fit(pm, ckpt_path=resume_from)

    print(f"\nDone: {model_name} | {mode_label}")
    print(f"  Best ckpt: {checkpoint_cb.best_model_path}\n")
    return checkpoint_cb.best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare eSEN / EquiformerV2 / ESCaIP on HORM"
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=MODELS + ['all'],
                        help='Model to train (default: all)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=MODES + ['all'],
                        help='Supervision mode: ef / efh / all (default: all)')
    parser.add_argument('--data', type=str, default='data/sample_100.lmdb',
                        help='Training LMDB')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Validation LMDB (defaults to --data)')
    parser.add_argument('--checkpoint', type=str,
                        default='ckpt/esen_sm_conserving_all.pt',
                        help='eSEN checkpoint (used for architecture config only)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None,
                        help='LR override (default: per-model HORM convention: '
                             'esen/eq2_cons/escaip=1e-4, equiformerv2=3e-4)')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--project', type=str,
                        default='horm-model-comparison')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Root dir for checkpoints and logs')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to last.ckpt to resume a single run')

    args = parser.parse_args()

    val_data = args.val_data or args.data

    models_to_run = MODELS if args.model == 'all' else [args.model]
    modes_to_run  = MODES  if args.mode  == 'all' else [args.mode]

    for model_key in models_to_run:
        for mode in modes_to_run:
            train_one(
                model_key=model_key,
                mode=mode,
                data_path=args.data,
                val_data_path=val_data,
                esen_checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
                project=args.project,
                devices=args.devices,
                output_dir=args.output_dir,
                use_wandb=not args.no_wandb,
                resume_from=args.resume if (len(models_to_run) == 1 and len(modes_to_run) == 1) else None,
            )


if __name__ == '__main__':
    main()
