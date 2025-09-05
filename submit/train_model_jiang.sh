#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=jiang
#SBATCH --time=150:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a6000:8
#SBATCH --cpus-per-task=48
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

export NUM_GPUS=8
export CONFIG_NAME=${SLURM_JOB_NAME}


echo $SLURM_JOB_NAME;
echo $1;


#module load cuda/11.3
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta2

WORKSPACE_PATH="/projects/vig/alberto/medical/deform-aneurysm-detection"



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
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir /dev/shm/internal_train 
cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/crop_0.4 /dev/shm/internal_train/crop_0.4

#cp -r  /scratch/ceballosarroyo.a/aneurysm/cta_datasets/internal_train/vein_mask_edt_comp /dev/shm/internal_train/vein_mask_edt_comp 
echo "Done copying" 


export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume\


