#!/bin/bash
#SBATCH --job-name=esen-horm-EFH
#SBATCH --output=/scratch/snirenbe/esen_horm_logs/esen_EFH_%j.out
#SBATCH --error=/scratch/snirenbe/esen_horm_logs/esen_EFH_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=simon_nirenberg@brown.edu
#SBATCH --mail-type=END
#SBATCH --account=rrg-aspuru

# eSEN from scratch — E+F+H (energy + autograd forces + stochastic autograd Hessian rows)
# Hessian requires create_graph=True — more memory than E/EF, so smaller batch size.

TRAIN_DATA="data/ts1x_hess_train.lmdb"
VAL_DATA="data/ts1x-val.lmdb"
CHECKPOINT="ckpt/esen_sm_conserving_all.pt"
BATCH_SIZE=32        # smaller than E/EF due to Hessian memory; reduce to 16 if OOM
MAX_EPOCHS=500
LEARNING_RATE=1e-4
PATIENCE=50
PROJECT_NAME="horm-esen-comparison-scratch"
OUTPUT_DIR="/scratch/snirenbe/esen_horm"

echo "================================================"
echo "eSEN HORM Training: E+F+H (from scratch)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "================================================"

module load StdEnv/2023 gcc/12.3 cuda/12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set cache dirs to writable locations (avoid permission issues on cluster)
export UV_CACHE_DIR=/scratch/snirenbe/.cache/uv
export MPLCONFIGDIR=/scratch/snirenbe/.cache/matplotlib
export XDG_CACHE_HOME=/scratch/snirenbe/.cache
export FAIRCHEM_CACHE_DIR=/scratch/snirenbe/.cache/fairchem
export WANDB_CACHE_DIR=/scratch/snirenbe/wandb
export WANDB_CONFIG_DIR=/scratch/snirenbe/wandb
export WANDB_DIR=/scratch/snirenbe/wandb
export WANDB_MODE=offline
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAIRCHEM_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR"

cd /project/rrg-aspuru/snirenbe/HORM || exit 1
source /project/rrg-aspuru/snirenbe/HORM/.venv/bin/activate

mkdir -p "$OUTPUT_DIR"

for f in "$TRAIN_DATA" "$VAL_DATA" "$CHECKPOINT"; do
    [ ! -e "$f" ] && echo "ERROR: not found: $f" && exit 1
done

# Auto-detect last.ckpt from a previous run to resume if job timed out.
RESUME_FLAG=""
LAST_CKPT=$(find "$OUTPUT_DIR/checkpoint/$PROJECT_NAME" -name "last.ckpt" -path "*eSEN-efh-scratch*" 2>/dev/null | sort | tail -1)
if [ -n "$LAST_CKPT" ]; then
    echo "Found previous checkpoint — resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
else
    echo "No previous checkpoint found — starting fresh."
fi

srun /project/rrg-aspuru/snirenbe/HORM/.venv/bin/python train_esen_comparison.py \
    --mode efh \
    --data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --project $PROJECT_NAME \
    --output_dir "$OUTPUT_DIR" \
    --devices 4 \
    --from_scratch \
    $RESUME_FLAG
STATUS=$?

DEST_DIR="/project/rrg-aspuru/snirenbe/HORM"
if [ $STATUS -eq 0 ]; then
    echo "Training complete — copying results to $DEST_DIR"
    rsync -av --ignore-existing \
        "$OUTPUT_DIR/checkpoint/$PROJECT_NAME/" \
        "$DEST_DIR/checkpoint/$PROJECT_NAME/"
    rsync -av --ignore-existing \
        "$OUTPUT_DIR/logs/$PROJECT_NAME/" \
        "$DEST_DIR/logs/$PROJECT_NAME/"
    echo "Done. Checkpoints and logs copied to project dir."
else
    echo "Job ended with status $STATUS (timeout or error) — not copying. Resubmit to resume."
fi
