#!/bin/bash
#SBATCH --job-name=IMMDF_CMNIST_DR
#SBATCH --array=0-49
#SBATCH --output=logs/cmnist_dr_%A_%a.out
#SBATCH --error=logs/cmnist_dr_%A_%a.err
#SBATCH --ntasks-per-node=1
# #SBATCH --partition=a100          # only if the queue is empty and you need the speed
#SBATCH --partition=gpu
# #SBATCH --gres=gpu
# #SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00

# ColorMNIST DR-CPME + IMQ MMD flow, one sweep axis x DGP seeds: array task ID =
# SEEDS_PER * value_index + data_seed. The default is SEEDS_PER=10 and a 5-value n_obs
# sweep, so 0-49 covers {2000, 5000, 10000, 15000, 20000} x 10 DGP seeds, each task
# running --n_repeats pipeline repeats (fresh split / AE / DR subsample / particles per
# repeat) on its fixed logged sample. Each repeat runs BOTH intervention arms on shared
# nuisances (one CSV row per arm; restrict with EXTRA="--target_arms 1"). Every task
# writes one self-describing CSV to results/ (rewritten after each repeat, so partial
# results survive a timeout).
#
# The sweep axis is generic: PARAM names the swept flag, VALUES its grid. Adjust
# --array at submit time to (#values x SEEDS_PER - 1). EXTRA appends fixed flags.
#
#   sbatch submit_colormnist_dr_mmd_flow_parallel.sh
#   PARAM="--n_particles" VALUES="200 400 800 1600 3200" TAG=Msweep \
#       sbatch --export=ALL submit_colormnist_dr_mmd_flow_parallel.sh
#   PARAM="--w_conf" VALUES="1 2 5 10 20" TAG=wsweep \
#       sbatch --export=ALL submit_colormnist_dr_mmd_flow_parallel.sh
#   PARAM="--alpha" VALUES="0 0.2" TAG=alphasweep \
#       sbatch --export=ALL --array=0-19 submit_colormnist_dr_mmd_flow_parallel.sh
#   PARAM="--k_mask" VALUES="10 20" TAG=ksweep \
#       sbatch --export=ALL --array=0-19 submit_colormnist_dr_mmd_flow_parallel.sh
#   TAG=w2ref EXTRA="--w_conf 2" sbatch --export=ALL submit_colormnist_dr_mmd_flow_parallel.sh

module load miniconda
module load cuda/13.0
PYTHON_EXEC="/nfs/ghome/live/bbozkurt/.conda/envs/torchnn/bin/python"

mkdir -p logs
echo "Checking GPU status..."
nvidia-smi

SEEDS_PER=${SEEDS_PER:-10}
PARAM="${PARAM:---n_obs}"
VALUES="${VALUES:-2000 5000 10000 15000 20000}"
read -r -a VALUE_ARR <<< "$VALUES"
VALUE=${VALUE_ARR[$((SLURM_ARRAY_TASK_ID / SEEDS_PER))]}
DATA_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS_PER))
TAG="${TAG:-refparams}"
EXTRA="${EXTRA:-}"

if [ -z "$VALUE" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds the sweep grid (${#VALUE_ARR[@]} values x $SEEDS_PER seeds)" >&2
    exit 1
fi

echo "Task ID: $SLURM_ARRAY_TASK_ID  ->  $PARAM $VALUE  data seed $DATA_SEED   tag $TAG   extra: $EXTRA"
echo "Environment: $PYTHON_EXEC"

cd "$SLURM_SUBMIT_DIR"
$PYTHON_EXEC colormnist_dr_mmd_flow_simulation.py \
    $PARAM $VALUE --data_seeds $DATA_SEED --n_repeats 3 --tag "$TAG" $EXTRA
