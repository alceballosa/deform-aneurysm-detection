#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=jiang
#SBATCH --time=150:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a5000:8
#SBATCH --cpus-per-task=72
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_JOB_NAME;
echo $1;

module unload cuda/12.1.1 && module load cuda/12.8.0
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta3
export WORKSPACE_PATH="/projects/vig/alberto/medical/exploration/deform"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH

export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$1/$2.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume
