#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=multigpu
#SBATCH --time=23:59:00
#SBATCH --mem=256
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --cpus-per-task=14
#SBATCH --output=./logs/exec.%j.%x_inf.out
#SBATCH --error=./logs/exec.%j.%x_inf.out
#SBATCH --nice=0
#SBATCH --exclude=d1026

export NUM_GPUS=2
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_JOB_NAME;

module load anaconda3/2022.05
#module load cuda/11.3
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
source activate cta


WORKSPACE_PATH="/home/ceballosarroyo.a/workspace/medical/cta-det2"


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
python src/train_net.py\
    --num-gpus ${NUM_GPUS}\
    --config-file "./configs/deform/${SLURM_JOB_NAME}.yaml"\
    --eval-only\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    MODEL.WEIGHTS /home/ceballosarroyo.a/workspace/medical/cta-det2/outputs/${SLURM_JOB_NAME}/model_$1.pth

python src/postprocess/csv_to_nifti.py --config-file "./configs/deform/${SLURM_JOB_NAME}.yaml" POSTPROCESS.CHECKPOINT "$1"