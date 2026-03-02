#!/bin/bash
#SBATCH --job-name=horm-comparison
#SBATCH --output=/scratch/snirenbe/esen_horm_logs/comparison_%j.out
#SBATCH --error=/scratch/snirenbe/esen_horm_logs/comparison_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=simon_nirenberg@brown.edu
#SBATCH --mail-type=END
#SBATCH --account=rrg-aspuru

# Model comparison on sample_100.lmdb:
#   4 models × 2 modes (EF, EFH) = 8 training runs, all sequential on 1 H100.
#   With 100 samples and 200 epochs each, all 8 runs finish well within 4h.
#
# Models:
#   esen                    — conserving eSEN (autograd forces)
#   equiformerv2            — EquiformerV2 with direct force head
#   equiformerv2_conservative — EquiformerV2 with autograd forces
#   escaip                  — ESCaIP with autograd forces
#
# To run a single (model, mode) pair:
#   sbatch --export=MODEL=esen,MODE=ef submit_comparison.sh

MODEL="${MODEL:-all}"
MODE="${MODE:-all}"

DATA="data/sample_100.lmdb"
CHECKPOINT="ckpt/esen_sm_conserving_all.pt"
BATCH_SIZE=16
MAX_EPOCHS=200
PATIENCE=30
PROJECT_NAME="horm-model-comparison"
OUTPUT_DIR="/scratch/snirenbe/esen_horm_comparison"

echo "=============================================="
echo "HORM Model Comparison"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Model(s): $MODEL  Mode(s): $MODE"
echo "=============================================="

module load StdEnv/2023 gcc/12.3 cuda/12.6

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export UV_CACHE_DIR=/scratch/snirenbe/.cache/uv
export MPLCONFIGDIR=/scratch/snirenbe/.cache/matplotlib
export XDG_CACHE_HOME=/scratch/snirenbe/.cache
export FAIRCHEM_CACHE_DIR=/scratch/snirenbe/.cache/fairchem
export WANDB_CACHE_DIR=/scratch/snirenbe/wandb
export WANDB_CONFIG_DIR=/scratch/snirenbe/wandb
export WANDB_DIR=/scratch/snirenbe/wandb
export WANDB_MODE=offline
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FAIRCHEM_CACHE_DIR" \
         "$WANDB_CACHE_DIR" "$WANDB_DIR" "$OUTPUT_DIR" \
         /scratch/snirenbe/esen_horm_logs

cd /project/rrg-aspuru/snirenbe/HORM || exit 1
source /project/rrg-aspuru/snirenbe/HORM/.venv/bin/activate

[ ! -e "$DATA" ]       && echo "ERROR: data not found: $DATA"             && exit 1
[ ! -e "$CHECKPOINT" ] && echo "ERROR: checkpoint not found: $CHECKPOINT" && exit 1

srun /project/rrg-aspuru/snirenbe/HORM/.venv/bin/python train_comparison.py \
    --model "$MODEL" \
    --mode  "$MODE"  \
    --data  "$DATA"  \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --project "$PROJECT_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --devices 1

STATUS=$?

DEST_DIR="/project/rrg-aspuru/snirenbe/HORM"
if [ $STATUS -eq 0 ]; then
    echo "All runs complete — copying results to $DEST_DIR"
    rsync -av --ignore-existing \
        "$OUTPUT_DIR/checkpoint/$PROJECT_NAME/" \
        "$DEST_DIR/checkpoint/$PROJECT_NAME/"
    rsync -av --ignore-existing \
        "$OUTPUT_DIR/logs/$PROJECT_NAME/" \
        "$DEST_DIR/logs/$PROJECT_NAME/"
    echo "Done."
else
    echo "Job ended with status $STATUS."
fi
