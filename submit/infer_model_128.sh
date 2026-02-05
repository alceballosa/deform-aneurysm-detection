#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=jiang
#SBATCH --time=24:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=32
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

export NUM_GPUS=2
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_GRES;
echo $SLURM_JOB_NAME;
echo $1;


#module load cuda/11.3
module unload cuda/12.1.1 && module load cuda/12.8.0
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta3

WORKSPACE_PATH="/projects/vig/alberto/medical/exploration/deform"



# export NCCL_P2P_LEVEL=PXB
if ((${NUM_GPUS} == 1)); then
    export CUDA_VISIBLE_DEVICES=0;
    echo "Using 1 GPU";
elif ((${NUM_GPUS} == 2)); then
    export CUDA_VISIBLE_DEVICES=0,1;
    echo "Using 2 GPUs";
elif ((${NUM_GPUS} == 4)); then
    export CUDA_VISIBLE_DEVICES=0,1,2,3;
    echo "Using 4 GPUs";
elif ((${NUM_GPUS} == 8)); then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
    echo "Using 8 GPUs";
fi;

export PYTHONPATH=${WORKSPACE_PATH}

export PYTHONPATH=$(pwd):$PYTHONPATH

./run_several_128.sh $1 $2 $NUM_GPUS

