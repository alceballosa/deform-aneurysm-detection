#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=ghx4
#SBATCH --account=bexf-dtai-gh
#SBATCH --time=24:00:00
#SBATCH --mem=512
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=10
#SBATCH --output=./logs/exec.%j.train_h100.%x.out
#SBATCH --error=./logs/exec.%j.train_h100.%x.out
#SBATCH --nice=0
#SBATCH --dependency=singleton




# export CONFIG_NAME=${SLURM_JOB_NAME}

# echo $SLURM_JOB_NAME;

# #module load cuda/11.3
# #source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
# #conda activate olmo2

# WORKSPACE_PATH="/u/aceballosarroyo/workspace/med_lm_training"
# cd $WORKSPACE_PATH
# #export PYTHONPATH=${WORKSPACE_PATH}

# #conda activate olmo2
# # ['configs/train_vcl_resnet_imagebd.yaml', 'configs/train_vcl_rn_bbox_imagebd.yaml']
# export PYTHONPATH=$(pwd):$PYTHONPATH


# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
# Model name from -J flag, fallback to $1 for local runs
export MODEL_NAME=${SLURM_JOB_NAME:-$1}

echo "Model: $MODEL_NAME"

#module unload cuda/12.1.1 && module load cuda/12.8.0
#source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh
#conda activate cta3
export WORKSPACE_PATH="/work/nvme/bexf/aceballosarroyo/deform"

cd $WORKSPACE_PATH

# Auto-detect config family from model name
source submit/find_family.sh "$MODEL_NAME"

export PYTHONPATH=${WORKSPACE_PATH}
export PYTHONPATH=$(pwd):$PYTHONPATH



# sleep 7m;

export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$FAMILY/$MODEL_NAME.yaml"\
        --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume\


