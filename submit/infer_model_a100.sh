#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=177huntington
#SBATCH --time=48:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a100:1
#SBATCH  -cpus-per-gpu=8
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
# Model name from -J flag, fallback to $1 for local runs
export MODEL_NAME=${SLURM_JOB_NAME:-$1}

echo "Model: $MODEL_NAME"

module unload cuda/12.1.1 && module load cuda/12.8.0
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta3
export WORKSPACE_PATH="/projects/vig/alberto/medical/exploration/deform"

cd $WORKSPACE_PATH

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/inference/run_evaluation.py --model $MODEL_NAME --num-gpus $NUM_GPUS
