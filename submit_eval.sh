#!/bin/bash
#SBATCH --job-name=esen-horm-eval
#SBATCH --output=logs/esen_eval_%j.out
#SBATCH --error=logs/esen_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Post-hoc evaluation of trained eSEN checkpoints on E, F, and H.
#
# Runs two evaluations matching HORM paper protocol:
#   1. In-distribution:      ts1x-val.lmdb  (same distribution as training)
#   2. Out-of-distribution:  RGD1.lmdb      (unseen reaction families)
#
# Full-Hessian eval (all 3N rows per molecule) requires GPU and is expensive.
# Use NUM_HESS_ROWS=N (e.g. 20) for a fast pass; leave empty for full eval.
#
# Usage:
#   sbatch submit_eval.sh
#   sbatch --export=RUN_DIR=checkpoint/my-run submit_eval.sh
#   sbatch --export=NUM_HESS_ROWS=20 submit_eval.sh   # fast mode

# ── Configuration ─────────────────────────────────────────────────────────────
# Checkpoints directory (containing e, ef, efh subdirs from training)
RUN_DIR="${RUN_DIR:-checkpoint/horm-esen-comparison-scratch}"

# In-distribution val set (HORM-Transition1x val, same distribution as training)
VAL_DATA="data/ts1x-val.lmdb"

# Out-of-distribution test set (HORM-RGD1, never seen during training)
RGD1_DATA="/project/rrg-aspuru/snirenbe/HORM/data/RGD1.lmdb"

# Hessian rows per molecule: empty = full (exact but slow), integer = sampled (fast)
NUM_HESS_ROWS="${NUM_HESS_ROWS:-}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================"
echo "eSEN HORM Evaluation"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Run dir: $RUN_DIR"
echo "================================================"

export PYTHONUNBUFFERED=1

cd /home/simonnir/esen_horm/HORM || exit 1
[ -f ".venv/bin/activate" ] && source .venv/bin/activate

mkdir -p results

[ ! -d "$RUN_DIR" ] && echo "ERROR: run dir not found: $RUN_DIR" && exit 1

HESS_FLAG=""
[ -n "$NUM_HESS_ROWS" ] && HESS_FLAG="--num_hessian_rows $NUM_HESS_ROWS"

# ── 1. In-distribution evaluation (ts1x-val) ─────────────────────────────────
echo ""
echo "=== In-distribution: ts1x-val ==="
if [ ! -e "$VAL_DATA" ]; then
    echo "WARNING: val data not found: $VAL_DATA -- skipping"
else
    python eval_trained.py \
        --run_dir "$RUN_DIR" \
        --data "$VAL_DATA" \
        --output "results/eval_indist.json" \
        $HESS_FLAG
fi

# ── 2. Out-of-distribution evaluation (RGD1) ─────────────────────────────────
echo ""
echo "=== Out-of-distribution: RGD1 ==="
if [ ! -e "$RGD1_DATA" ]; then
    echo "WARNING: RGD1 data not found: $RGD1_DATA -- skipping"
else
    python eval_trained.py \
        --run_dir "$RUN_DIR" \
        --data "$RGD1_DATA" \
        --output "results/eval_ood_rgd1.json" \
        $HESS_FLAG
fi

echo ""
echo "================================================"
echo "Evaluation complete."
echo "  In-distribution: results/eval_indist.json"
echo "  OOD (RGD1):      results/eval_ood_rgd1.json"
echo "================================================"
