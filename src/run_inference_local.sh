#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=$4 python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/deform/$1.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS ./models/$1/model_$2.pth

python src/postprocess/csv_to_nifti.py --config-file "./configs/deform/$1.yaml" POSTPROCESS.CHECKPOINT "$2" POSTPROCESS.THRESHOLD "$3"

