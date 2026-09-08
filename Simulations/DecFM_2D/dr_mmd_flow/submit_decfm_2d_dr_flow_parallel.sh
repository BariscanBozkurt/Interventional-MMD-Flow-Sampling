#!/bin/bash
#SBATCH --job-name=IMMDF_2D_DR
#SBATCH --array=0-49
#SBATCH --output=logs/decfm2d_dr_%A_%a.out
#SBATCH --error=logs/decfm2d_dr_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu             # CPU-only experiment (validated with CUDA hidden);
                                    # adjust to this cluster's CPU partition name, or
                                    # fall back to gpu_lowp + --gres=gpu:1 if none fits
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00

# DecFM 2D mixture, DR-CPME target, BOTH samplers (RBF forward-Euler + Riesz AdamW)
# run per repeat from the same paired initial particles, both arms. One sweep axis x
# DGP seeds: array task ID = SEEDS_PER * value_index + data_seed. The default is
# SEEDS_PER=10 and the ColorMNIST-compatible n_obs grid, so 0-49 covers
# {2000, 5000, 10000, 15000, 20000} x 10 DGP seeds, each task running --n_repeats
# repeats -> 12 CSV rows per repeat task (3 repeats x 2 arms x 2 samplers... 12 rows
# per task with the default 3 repeats: 3 x 2 x 2). Every task writes one
# self-describing CSV to results/ (rewritten after each repeat).
#
# 20G covers the n = 20000 tasks (measured peak RSS 11.4 GB): the outcome-kernel median lock and the unbiased
# MMD^2 against the full oracle draw each materialise a 20000^2 float64 matrix
# (~3.2 GB) transiently. The DR support is capped at --n_dr 4000 (identity order at
# n_obs <= 4000, the notebooks' exact path).
#
#   sbatch submit_decfm_2d_dr_flow_parallel.sh
#   PARAM="--n_particles" VALUES="250 500 1000 2000 4000" TAG=Msweep \
#       sbatch --export=ALL submit_decfm_2d_dr_flow_parallel.sh
#   PARAM="--confounding_strength" VALUES="1 2 3 5 8" TAG=gammasweep \
#       sbatch --export=ALL submit_decfm_2d_dr_flow_parallel.sh
#   PARAM="--mixture_delta" VALUES="1 2 4" TAG=deltasweep \
#       sbatch --export=ALL --array=0-29 submit_decfm_2d_dr_flow_parallel.sh
#   TAG=eulertuned EXTRA="--samplers rbf_euler --step_factor 0.15" \
#       sbatch --export=ALL submit_decfm_2d_dr_flow_parallel.sh

module load miniconda
PYTHON_EXEC="/nfs/ghome/live/bbozkurt/.conda/envs/torchnn/bin/python"

mkdir -p logs

SEEDS_PER=${SEEDS_PER:-10}
PARAM="${PARAM:---n_obs}"
VALUES="${VALUES:-2000 5000 10000 15000 20000}"
read -r -a VALUE_ARR <<< "$VALUES"
VALUE=${VALUE_ARR[$((SLURM_ARRAY_TASK_ID / SEEDS_PER))]}
DATA_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS_PER))
TAG="${TAG:-twins}"
EXTRA="${EXTRA:-}"

if [ -z "$VALUE" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds the sweep grid (${#VALUE_ARR[@]} values x $SEEDS_PER seeds)" >&2
    exit 1
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID  ->  $PARAM $VALUE  data seed $DATA_SEED   tag $TAG   extra: $EXTRA"
echo "Environment: $PYTHON_EXEC"

cd "$SLURM_SUBMIT_DIR"
$PYTHON_EXEC decfm_2d_dr_flow_simulation.py \
    $PARAM $VALUE --data_seeds $DATA_SEED --n_repeats 3 --tag "$TAG" $EXTRA
