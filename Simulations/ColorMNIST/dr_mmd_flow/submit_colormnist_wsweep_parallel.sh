#!/bin/bash
#SBATCH --job-name=IMMDF_CMNIST_W
#SBATCH --array=0-29
#SBATCH --output=logs/cmnist_w_%A_%a.out
#SBATCH --error=logs/cmnist_w_%A_%a.err
#SBATCH --ntasks-per-node=1
# #SBATCH --partition=a100          # only if the queue is empty and you need the speed
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00

# ColorMNIST CONFOUNDING-STRENGTH ablation (the Figure-3 experiment of Dance et al.,
# 2026): sweep w in e(1|x) = sigmoid(w (x - 1/2)) from 1 to 10 at n_obs = 20000, all
# other parameters at the reference values (alpha = 0, k = 10, tau = 0.08, digits
# (1,6) — the doflow code's settings). Array task ID = 3 * w_index + data_seed, so
# 0-29 covers w in {1..10} x 3 DGP seeds (the paper's seed count), each task running
# --n_repeats pipeline repeats x BOTH arms on shared nuisances. The Figure-3 curve is
# then avg-over-arms sw2_flow vs w, with sw2_initial as the no-deconfounding
# baseline and sw2_floor as each draw's oracle-vs-oracle floor.
#
# Two knowing differences from the paper's exact evaluation, recorded here so the
# comparison is honest: (i) our references are fresh oracle draws of size M = 1600
# (matched-n protocol) where the paper scores against 5000 references — SW2 LEVELS
# shift with reference size, shapes are comparable; (ii) the paper's DGP clamps the
# true propensity to [0.01, 0.99], ours does not — the clamp only binds at w = 10
# and moves arm assignment for <1% of x, negligible but nonzero.
#
#   sbatch submit_colormnist_wsweep_parallel.sh
#   VALUES="1 2 5 10" TAG=wsweep_coarse \
#       sbatch --export=ALL --array=0-11 submit_colormnist_wsweep_parallel.sh

module load miniconda
module load cuda/13.0
PYTHON_EXEC="/nfs/ghome/live/bbozkurt/.conda/envs/torchnn/bin/python"

mkdir -p logs
echo "Checking GPU status..."
nvidia-smi

SEEDS_PER=${SEEDS_PER:-3}
VALUES="${VALUES:-1 2 3 4 5 6 7 8 9 10}"
read -r -a VALUE_ARR <<< "$VALUES"
W=${VALUE_ARR[$((SLURM_ARRAY_TASK_ID / SEEDS_PER))]}
DATA_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS_PER))
TAG="${TAG:-wsweep}"
EXTRA="${EXTRA:-}"

if [ -z "$W" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds the sweep grid (${#VALUE_ARR[@]} values x $SEEDS_PER seeds)" >&2
    exit 1
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID  ->  --w_conf $W  data seed $DATA_SEED   tag $TAG   extra: $EXTRA"
echo "Environment: $PYTHON_EXEC"

cd "$SLURM_SUBMIT_DIR"
$PYTHON_EXEC colormnist_dr_mmd_flow_simulation.py \
    --w_conf $W --n_obs 20000 --data_seeds $DATA_SEED --n_repeats 3 --tag "$TAG" $EXTRA
