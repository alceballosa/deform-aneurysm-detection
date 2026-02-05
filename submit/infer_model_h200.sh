#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=multigpu
#SBATCH --time=24:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_GRES;
echo $SLURM_JOB_NAME;
echo $1;

module unload cuda/12.1.1 && module load cuda/12.8.0
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta3
export WORKSPACE_PATH="/projects/vig/alberto/medical/exploration/deform"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH

python scripts/inference/run_evaluation.py --family $1 --model $2 --num-gpus $NUM_GPUS
