#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=177huntington
#SBATCH --time=72:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a100:2
#SBATCH  -cpus-per-gpu=8
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0

# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
export CONFIG_NAME=${SLURM_JOB_NAME}

echo $SLURM_JOB_NAME;

module unload cuda/12.1.1 && module load cuda/12.8.0
source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate cta3
export WORKSPACE_PATH="/projects/vig/alberto/medical/exploration/deform"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH



#mkdir /dev/shm/internal_train 

#if [ ! -d "/dev/shm/internal_train/crop_0.4" ]; then
    #echo "Copying data"
#    cp -r  /projects/vig/Datasets/aneurysm/cta_datasets/internal_train/crop_0.4  /dev/shm/internal_train/crop_0.4
    #echo "Copying data for vessels"
    #cp -r  /projects/vig/Datasets/aneurysm/cta_datasets/internal_train/crop_0.4_vessel_edt_comp /dev/shm/internal_train/crop_0.4_vessel_edt_comp
    #cp -r  /projects/vig/Datasets/aneurysm/cta_datasets/internal_train/vein_mask_edt_comp /dev/shm/internal_train/vein_mask_edt_comp 
    #echo "Done copying" 
#fi

./submit/train/run_2_gpu.sh $1 $2

