#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta2 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
export ROOT_DATA="/scratch/ceballosarroyo.a/aneurysm/cta_datasets/"
CUDA_VISIBLE_DEVICES=$5 python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/$1/$2.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $3\
    DATA.DIR.VAL.SCAN_DIR "${ROOT_DATA}/external/crop_0.4"\
    DATA.DIR.VAL.ANNOTATION_FILE "./labels/gt/internal_test_crop_0.4.csv"\
    DATA.DIR.VAL.VESSEL_DIR "${ROOT_DATA}/external/crop_0.4_vessel_edt_comp"\
    DATA.DIR.VAL.CVS_DIR "${ROOT_DATA}/external/vein_mask_edt_comp"
