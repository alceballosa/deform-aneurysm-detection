#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=multigpu
#SBATCH --time=23:59:00
#SBATCH --mem=96
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --output=./logs/exec.%j.%x_inf.out
#SBATCH --error=./logs/exec.%j.%x_inf.out
#SBATCH --nice=0


export NUM_GPUS=1
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_JOB_NAME;


source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta2

WORKSPACE_PATH="/home/ceballosarroyo.a/workspace/medical/deform-aneurysm-detection"

cd $WORKSPACE_PATH

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

# ['configs/train_vcl_resnet_imagebd.yaml', 'configs/train_vcl_rn_bbox_imagebd.yaml']

export ID_PORT=$(($RANDOM+20000))

export PYTHONPATH=$(pwd):$PYTHONPATH

./run_several_trx_vein.sh 

# TODO: fix a bunch of stuff