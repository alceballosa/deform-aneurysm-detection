#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 
export ID_PORT=$(($RANDOM+20010))
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_net.py\
    --num-gpus 4\
    --config-file "./configs/deform/$1.yaml"\
    --eval-only\
    MODEL.WEIGHTS "./models/$1/model_$2.pth" \
    DATA.DIR.VAL.SCAN_DIR $3\
    DATA.DIR.VAL.VESSEL_DIR $4\
    OUTPUT_DIR $5



python src/postprocess/csv_to_nifti.py --config-file "./configs/deform/$1.yaml" POSTPROCESS.CHECKPOINT "$2" POSTPROCESS.THRESHOLD $6 OUTPUT_DIR $5 DATA.DIR.VAL.SCAN_DIR $3
