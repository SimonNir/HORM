"""
Fine-tuning script for eSEN on HORM dataset.

This script fine-tunes the pretrained eSEN model on HORM data with:
- Energy + Force supervision (E+F)
- Energy + Force + Hessian supervision (E+F+H)
"""

from uuid import uuid4
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from training_module import PotentialModule


torch.set_float32_matmul_precision('high')

# Configuration
model_type = "eSEN"
version = "ef"  # Change to "efh" for E+F+H training
project = "horm-esen"
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

model_config = dict(
    name="eSEN",
    checkpoint_path="ckpt/esen_sm_conserving_all.pt",  # or esen_sm_conserving_all.pt
    device="cuda",
    num_targets=1,
    output_dim=1,
    readout="sum",
    use_pbc=False,
    direct_forces=False,  # eSEN uses autograd for forces
    pos_require_grad=True,
    compute_forces=True,
    compute_stress=False,
)

optimizer_config = dict(
    lr=1e-4,  # Lower LR for fine-tuning
    betas=[0.9, 0.999],
    weight_decay=1e-5,
    amsgrad=True,
)

training_config = dict(
    trn_path="data/sample_100.lmdb",
    val_path="data/sample_100.lmdb",
    bz=2,
    num_workers=8,
    clip_grad=True,
    gradient_clip_val=0.1,
    ema=False,
    lr_schedule_type="step",
    lr_schedule_config=dict(
        gamma=0.9,
        step_size=100,
    ),
)

# Initialize model
pm = PotentialModule(model_config, optimizer_config, training_config)

# Logger
logger = WandbLogger(
    project=project,
    log_model=False,
    name=run_name,
)

ckpt_path = f"checkpoint/{project}/{logger.experiment.name}"

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="esen-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}",
    every_n_epochs=10,
    save_top_k=3,
)

early_stopping_callback = EarlyStopping(
    monitor="val-totloss",
    patience=200,
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
    devices=1,
    num_nodes=1,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    max_epochs=1000,
    callbacks=callbacks,
    default_root_dir=ckpt_path,
    logger=logger,
    gradient_clip_val=0.1,
    accumulate_grad_batches=1,
    limit_train_batches=1600,
    limit_val_batches=80,
)

# Train
print(f"\n{'='*60}")
print(f"Fine-tuning eSEN on HORM dataset")
print(f"{'='*60}")
print(f"Model: {model_config['name']}")
print(f"Checkpoint: {model_config['checkpoint_path']}")
print(f"Version: {version}")
print(f"Project: {project}")
print(f"Run name: {run_name}")
print(f"{'='*60}\n")

trainer.fit(pm)
