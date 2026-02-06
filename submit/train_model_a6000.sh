#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ceballosarroyo.a@northeastern.edu
#SBATCH --nodes=1
#SBATCH --partition=jiang
#SBATCH --time=150:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-gpu=8
#SBATCH --output=./logs/exec.%j.%x.out
#SBATCH --error=./logs/exec.%j.%x.out
#SBATCH --nice=0
#SBATCH --dependency=singleton

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

# Auto-detect config family from model name
source submit/find_family.sh "$MODEL_NAME"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH

#mkdir /dev/shm/internal_train
#cp -r  /projects/vig/Datasets/aneurysm/cta_datasets/internal_train/crop_0.4 /dev/shm/internal_train/crop_0.4

#cp -r  /projects/vig/Datasets/aneurysm/cta_datasets/internal_train/vein_mask_edt_comp /dev/shm/internal_train/vein_mask_edt_comp
#echo "Done copying"

export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$FAMILY/$MODEL_NAME.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume
