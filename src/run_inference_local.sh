#!/bin/bash

conda activate cta2

cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python src/train_net.py\
    --num-gpus 1\
    --config-file "/workspace/deform-aneurysm-detection/configs/docker_inference/$1.yaml"\
    --eval-only\
    MODEL.WEIGHTS /workspace/inputs/models/$1/model_$2.pth

python src/postprocess/csv_to_nifti.py --config-file "./configs/docker_inference/$1.yaml" POSTPROCESS.CHECKPOINT "$2" POSTPROCESS.THRESHOLD "$3"

