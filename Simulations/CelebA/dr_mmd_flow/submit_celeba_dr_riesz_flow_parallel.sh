#!/bin/bash
#SBATCH --job-name=IMMDF_CELEBA_DR
#SBATCH --array=0-8
#SBATCH --output=logs/celeba_dr_%A_%a.out
#SBATCH --error=logs/celeba_dr_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100          # only if the queue is empty and you need the speed
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu
# #SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# CelebA DR-CPME + Riesz MMD flow (AdamW, warm-up + cosine), SD-VAE latents, one
# sweep axis x data seeds: array task ID = SEEDS_PER * value_index + data_seed.
# CelebA defaults are heavier than the other experiments, so SEEDS_PER=3 and a
# 3-value n_obs sweep at 256 px: 0-8 covers {1024, 2048, 4096} x 3 data seeds, each
# task running --n_repeats repeats x BOTH arms on shared encodings and nuisances
# (restrict with EXTRA="--target_arms 0"). Every task writes one self-describing CSV
# to results/ (rewritten after each repeat, so partial results survive a timeout).
#
# Data caches (one .pt per image_size/n_obs/n_ref/seed) are generated on first use
# from the raw CelebA root on NFS and written atomically, so concurrent tasks that
# share a configuration (e.g. a particle sweep) cannot corrupt them — at worst the
# first wave duplicates one generation. MEMORY: 24G fits 256 px up to n_obs ~4096;
# n_obs = 20000 needs --image_size 64 (that IS the DecFM paper's setting — the
# head-to-head with its Figure 4 pending in CLAUDE.md):
#
#   sbatch submit_celeba_dr_riesz_flow_parallel.sh
#   # 256 px sweep with node-local caches on a100 (no home quota needed):
#   CACHE_DIR=local TAG=riesz sbatch --export=ALL --partition=a100 --mem=48G \
#       --array=0-2,4-8 submit_celeba_dr_riesz_flow_parallel.sh
#   PARAM="--n_obs" VALUES="20000" TAG=decfm64 EXTRA="--image_size 64" \
#       sbatch --export=ALL --array=0-2 submit_celeba_dr_riesz_flow_parallel.sh
#   # n=16384 at 256 px (the n16384 notebooks' setting): ~16 GB of data, so override
#   # the memory at submit time; --eval_max bounds the observed/round-trip evaluation
#   # (the ~8800-row arm would otherwise need ~7 GB of GPU for the pixel matrices)
#   PARAM="--n_obs" VALUES="16384" TAG=n16k EXTRA="--eval_max 2048" \
#       sbatch --export=ALL --array=0-2 --mem=64G submit_celeba_dr_riesz_flow_parallel.sh
#   # matched-M control for the male n16384 confound (pending in CLAUDE.md):
#   PARAM="--n_particles" VALUES="922 1024" TAG=maleM \
#       EXTRA="--n_obs 16384 --target_arms 1 --eval_max 2048" \
#       sbatch --export=ALL --array=0-5 --mem=64G submit_celeba_dr_riesz_flow_parallel.sh
#   PARAM="--n_particles" VALUES="256 512 1024 2048" TAG=Msweep \
#       sbatch --export=ALL --array=0-11 submit_celeba_dr_riesz_flow_parallel.sh
#   PARAM="--learning_rate" VALUES="0.0003 0.001 0.003" TAG=lrsweep \
#       sbatch --export=ALL submit_celeba_dr_riesz_flow_parallel.sh
#   PARAM="--flow_iterations" VALUES="2000 4000 8000" TAG=itersweep \
#       sbatch --export=ALL submit_celeba_dr_riesz_flow_parallel.sh
#   TAG=const EXTRA="--lr_schedule const" sbatch --export=ALL submit_celeba_dr_riesz_flow_parallel.sh

module load miniconda
module load cuda/13.0
PYTHON_EXEC="/nfs/ghome/live/bbozkurt/.conda/envs/torchnn/bin/python"

# The SD-VAE weights are already in the NFS-shared HF cache; forbid network lookups
# so an offline compute node cannot hang on a hub etag check.
export HF_HUB_OFFLINE=1

mkdir -p logs
echo "Checking GPU status..."
nvidia-smi

# Cache location: the default (repo data/celeba_dgp) lives on the HOME quota, and
# the full 256 px sweep needs ~40 GB of caches — the first TWO launches DIED ON
# EDQUOT there. Three ways to run:
#   (a) free home space and use the default (caches persist and are shared);
#   (b) CACHE_DIR=/some/filesystem/with/room (persistent, shared);
#   (c) CACHE_DIR=local — each task generates its cache on the COMPUTE NODE's local
#       scratch and deletes it on exit. Costs a few minutes of generation per task,
#       needs ZERO home quota, nothing is shared or persisted. Prefer a100 nodes for
#       this (large local disk); if the node's /tmp is RAM-backed tmpfs the cache
#       counts against job memory, so pair it with a generous --mem.
CACHE_DIR="${CACHE_DIR:-}"
if [ "$CACHE_DIR" = "local" ]; then
    CACHE_DIR="${SLURM_TMPDIR:-/tmp}/celeba_dgp_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$CACHE_DIR"
    trap 'rm -rf "$CACHE_DIR"' EXIT
    echo "node-local cache dir: $CACHE_DIR (deleted on exit)"
    df -h "$CACHE_DIR" | tail -1
fi
if [ -n "$CACHE_DIR" ]; then
    EXTRA="${EXTRA:-} --cache_dir $CACHE_DIR"
fi

SEEDS_PER=${SEEDS_PER:-3}
PARAM="${PARAM:---n_obs}"
VALUES="${VALUES:-1024 2048 4096}"
read -r -a VALUE_ARR <<< "$VALUES"
VALUE=${VALUE_ARR[$((SLURM_ARRAY_TASK_ID / SEEDS_PER))]}
DATA_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS_PER))
TAG="${TAG:-riesz}"
EXTRA="${EXTRA:-}"

if [ -z "$VALUE" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds the sweep grid (${#VALUE_ARR[@]} values x $SEEDS_PER seeds)" >&2
    exit 1
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID  ->  $PARAM $VALUE  data seed $DATA_SEED   tag $TAG   extra: $EXTRA"
echo "Environment: $PYTHON_EXEC"

cd "$SLURM_SUBMIT_DIR"
$PYTHON_EXEC celeba_dr_riesz_flow_simulation.py \
    $PARAM $VALUE --data_seeds $DATA_SEED --n_repeats 3 --tag "$TAG" $EXTRA
