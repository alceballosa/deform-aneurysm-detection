#!/usr/bin/env bash
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=ghx4
#SBATCH --account=bexf-dtai-gh
#SBATCH --time=48:00:00
#SBATCH --mem=120g
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=./logs/exec.%j.train_h100.%x.out
#SBATCH --error=./logs/exec.%j.train_h100.%x.out
#SBATCH --nice=0
#SBATCH --dependency=singleton



# Auto-detect number of GPUs from SLURM allocation
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
# Model name from -J flag, fallback to $1 for local runs
export MODEL_NAME=${SLURM_JOB_NAME:-$1}

echo "Model: $MODEL_NAME"


module load cuda/12.4.0

conda activate cta3
export WORKSPACE_PATH="/work/nvme/bexf/aceballosarroyo/deform"
which python
cd $WORKSPACE_PATH

# Auto-detect config family from model name
source submit/find_family.sh "$MODEL_NAME"

export PYTHONPATH=${WORKSPACE_PATH}
#export PYTHONPATH=$(pwd):$PYTHONPATH


export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus $NUM_GPUS\
    --config-file "./configs/$FAMILY/$MODEL_NAME.yaml"\
        --dist-url "tcp://127.0.0.1:20000"\
    --resume\


