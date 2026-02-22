#!/bin/bash
#SBATCH --job-name=esen-horm-EFH
#SBATCH --output=logs/esen_EFH_%j.out
#SBATCH --error=logs/esen_EFH_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# eSEN from scratch — E+F+H (energy + autograd forces + stochastic autograd Hessian rows)
# NHR=1: one Hessian row per molecule per step (standard for conserving/autograd models)
# Batch size reduced vs E/EF because Hessian requires create_graph=True (more memory)

TRAIN_DATA="data/ts1x_hess_train.lmdb"
VAL_DATA="data/ts1x-val.lmdb"
CHECKPOINT="ckpt/esen_sm_conserving_all.pt"
BATCH_SIZE=16        # reduce to 8 if OOM
MAX_EPOCHS=500
LEARNING_RATE=1e-4
PATIENCE=50
PROJECT_NAME="horm-esen-comparison-scratch"

echo "================================================"
echo "eSEN HORM Training: E+F+H (from scratch)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/simonnir/esen_horm/HORM || exit 1
[ -f ".venv/bin/activate" ] && source .venv/bin/activate

mkdir -p logs checkpoint

for f in "$TRAIN_DATA" "$VAL_DATA" "$CHECKPOINT"; do
    [ ! -e "$f" ] && echo "ERROR: not found: $f" && exit 1
done

# Auto-detect last.ckpt from a previous run to resume if job timed out.
RESUME_FLAG=""
LAST_CKPT=$(find "checkpoint/$PROJECT_NAME" -name "last.ckpt" -path "*eSEN-efh-scratch*" 2>/dev/null | sort | tail -1)
if [ -n "$LAST_CKPT" ]; then
    echo "Found previous checkpoint — resuming from: $LAST_CKPT"
    RESUME_FLAG="--resume $LAST_CKPT"
else
    echo "No previous checkpoint found — starting fresh."
fi

python train_esen_comparison.py \
    --mode efh \
    --data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --project $PROJECT_NAME \
    --devices 1 \
    --from_scratch \
    $RESUME_FLAG
