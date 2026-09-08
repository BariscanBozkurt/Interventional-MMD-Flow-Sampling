#!/bin/bash
#SBATCH --job-name=IMMDF_DSPR_DR
#SBATCH --array=0-49
#SBATCH --output=logs/dsprites_dr_%A_%a.out
#SBATCH --error=logs/dsprites_dr_%A_%a.err
#SBATCH --ntasks-per-node=1
# #SBATCH --partition=a100          # only if the queue is empty and you need the speed
#SBATCH --partition=gpu
# #SBATCH --gres=gpu
# #SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00

# dSprites continuous-action DR-CPME + Riesz MMD flow (AdamW), disjoint autoencoder
# split, one sweep axis x DGP seeds: array task ID = SEEDS_PER * value_index +
# data_seed. The default is SEEDS_PER=10 and a 5-value n_obs sweep, so 0-49 covers
# {500, 1000, 2000, 4000, 8000} x 10 DGP seeds (the causal half is n_obs/2; the
# reference notebook's causal size of 1000 corresponds to n_obs=2000), each task
# running --n_repeats pipeline repeats (fresh split / AE / nuisances / flow per
# repeat) on its fixed logged sample. Every task writes one self-describing CSV to
# results/ (rewritten after each repeat, so partial results survive a timeout).
#
# The sweep axis is generic: PARAM names the swept flag, VALUES its grid. Adjust
# --array at submit time to (#values x SEEDS_PER - 1). EXTRA appends fixed flags.
#
#   sbatch submit_dsprites_dr_riesz_flow_parallel.sh
#   PARAM="--policy_shift" VALUES="0.05 0.1 0.2 0.4 0.8" TAG=shiftsweep \
#       sbatch --export=ALL submit_dsprites_dr_riesz_flow_parallel.sh
#   PARAM="--sigma_policy" VALUES="0.25 0.5 1.0" TAG=overlapsweep \
#       sbatch --export=ALL --array=0-29 submit_dsprites_dr_riesz_flow_parallel.sh
#   PARAM="--pixel_noise" VALUES="0.02 0.05 0.1 0.2" TAG=noisesweep \
#       sbatch --export=ALL --array=0-39 submit_dsprites_dr_riesz_flow_parallel.sh
#   PARAM="--learning_rate" VALUES="0.003 0.01 0.03" TAG=lrsweep \
#       sbatch --export=ALL --array=0-29 submit_dsprites_dr_riesz_flow_parallel.sh
#   TAG=bigflow EXTRA="--flow_iterations 4000" sbatch --export=ALL submit_dsprites_dr_riesz_flow_parallel.sh
#   TAG=noclip EXTRA="--weight_clip 0" sbatch --export=ALL submit_dsprites_dr_riesz_flow_parallel.sh   # pre-2026-09 unclipped behaviour

module load miniconda
module load cuda/13.0
PYTHON_EXEC="/nfs/ghome/live/bbozkurt/.conda/envs/torchnn/bin/python"

mkdir -p logs
echo "Checking GPU status..."
nvidia-smi

SEEDS_PER=${SEEDS_PER:-10}
PARAM="${PARAM:---n_obs}"
VALUES="${VALUES:-500 1000 2000 4000 8000}"
read -r -a VALUE_ARR <<< "$VALUES"
VALUE=${VALUE_ARR[$((SLURM_ARRAY_TASK_ID / SEEDS_PER))]}
DATA_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS_PER))
TAG="${TAG:-splitae}"
EXTRA="${EXTRA:-}"

if [ -z "$VALUE" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds the sweep grid (${#VALUE_ARR[@]} values x $SEEDS_PER seeds)" >&2
    exit 1
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID  ->  $PARAM $VALUE  data seed $DATA_SEED   tag $TAG   extra: $EXTRA"
echo "Environment: $PYTHON_EXEC"

cd "$SLURM_SUBMIT_DIR"
$PYTHON_EXEC dsprites_dr_riesz_flow_simulation.py \
    $PARAM $VALUE --data_seeds $DATA_SEED --n_repeats 3 --tag "$TAG" $EXTRA
