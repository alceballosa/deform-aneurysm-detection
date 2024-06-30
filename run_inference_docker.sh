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

# docker command

#

# sudo docker run --gpus all -it --rm -v /{OUTPUT_PATH}/:/workspace/cta-det2/outputs/  -v /{INPUT_FOLDER}/:/data/ --shm-size=32g --ulimit memlock=-1 cta-detection2:latest ./cta-det2/run_inference_docker.sh "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_PRIV" "final"
