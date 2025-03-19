#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=$5 python src/train_net.py\
    --num-gpus 4\
    --config-file "./configs/$1/$2.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $3\
    DATA.DIR.VAL.SCAN_DIR "/data/aneurysm/internal_test/crop_0.4"\
    DATA.DIR.VAL.ANNOTATION_FILE "./labels/gt/internal_test_crop_0.4.csv"\
    DATA.DIR.VAL.VESSEL_DIR "/data/aneurysm/internal_test/crop_0.4_vessel_edt_comp"
