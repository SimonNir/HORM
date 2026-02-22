#!/bin/bash
#SBATCH --job-name=esen-horm-comparison
#SBATCH --output=logs/esen_comparison_%j.out
#SBATCH --error=logs/esen_comparison_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ESEN Training Comparison on HORM Dataset
# Trains eSEN from SCRATCH (random init, not from OMol25 pretrained weights)
# with three supervision levels:
#   - E     (energy only)
#   - E+F   (energy + autograd forces)
#   - E+F+H (energy + autograd forces + autograd hessians)
#
# Key scientific question:
#   Does training on Hessians (second-order autograd) improve Hessian prediction
#   compared to training only on E+F?
#
# Run as:
#   sbatch submit_esen_comparison.sh            # all three variants sequentially
#   sbatch --export=MODE=ef submit_esen_comparison.sh  # single variant

echo "============================================"
echo "eSEN HORM Training Comparison (From Scratch)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Date:   $(date)"
echo "============================================"
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
# Production HORM dataset paths (on cluster)
TRAIN_DATA="data/ts1x_hess_train.lmdb"
VAL_DATA="data/ts1x-val.lmdb"

# Architecture source: the script reads config from this checkpoint but
# does NOT load the weights (FROM_SCRATCH=true overrides that)
CHECKPOINT="ckpt/esen_sm_conserving_all.pt"

BATCH_SIZE=16        # per-GPU; reduce to 8 if OOM with EFH
MAX_EPOCHS=500
LEARNING_RATE=1e-4
PATIENCE=50          # early stopping
PROJECT_NAME="horm-esen-comparison-scratch"

# Training mode: 'all' runs e, ef, efh sequentially in one job.
# Pass MODE env var to override: sbatch --export=MODE=ef submit_esen_comparison.sh
TRAINING_MODE="${MODE:-all}"

USE_WANDB=true
FROM_SCRATCH=true    # MUST be true for these experiments
# ─────────────────────────────────────────────────────────────────────────────

echo "Configuration:"
echo "  Train data: $TRAIN_DATA"
echo "  Val data:   $VAL_DATA"
echo "  Checkpoint: $CHECKPOINT (config only)"
echo "  Mode:       $TRAINING_MODE"
echo "  From scratch: $FROM_SCRATCH"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Patience:   $PATIENCE"
echo "  WandB:      $USE_WANDB"
echo ""

# ── Environment setup ────────────────────────────────────────────────────────
# Uncomment and adapt for your cluster:
# module load cuda/12.1
# module load python/3.10
# source /path/to/venv/bin/activate   # or: conda activate myenv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the HORM working directory
cd /home/simonnir/esen_horm/HORM || exit 1

# Activate the local venv (created during setup)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create output directories
mkdir -p logs checkpoint

# Verify data files exist
if [ ! -f "$TRAIN_DATA" ] && [ ! -d "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    exit 1
fi
if [ ! -f "$VAL_DATA" ] && [ ! -d "$VAL_DATA" ]; then
    echo "ERROR: Validation data not found: $VAL_DATA"
    exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting training..."
echo ""

CMD="python train_esen_comparison.py \
    --mode $TRAINING_MODE \
    --data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --checkpoint $CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --project $PROJECT_NAME \
    --devices 1"

[ "$FROM_SCRATCH" = true ] && CMD="$CMD --from_scratch"
[ "$USE_WANDB" = false ]   && CMD="$CMD --no_wandb"

echo "Executing: $CMD"
echo ""
eval $CMD
STATUS=$?

echo ""
echo "============================================"
if [ $STATUS -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training FAILED with exit code $STATUS"
fi
echo "Checkpoints: checkpoint/$PROJECT_NAME/"
echo "Logs:        logs/$PROJECT_NAME/"
echo "============================================"

exit $STATUS
