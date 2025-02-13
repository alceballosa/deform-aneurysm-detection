#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=$5 python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/$1/$2.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $3

python src/postprocess/csv_to_nifti.py --config-file "./configs/$1/$2.yaml" POSTPROCESS.CHECKPOINT "$3" POSTPROCESS.THRESHOLD "$4"

