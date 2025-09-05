#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta2 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH



python src/train_net.py\
    --num-gpus $6\
    --config-file "./configs/$2/$3.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $4\
    DATA.DIR.VAL.SCAN_DIR "$1/crop_0.4"\
    DATA.DIR.VAL.ANNOTATION_FILE "./labels/gt/internal_test_crop_0.4.csv"\
    DATA.DIR.VAL.VESSEL_DIR "$1/crop_0.4_vessel_edt_comp"\
    DATA.DIR.VAL.CVS_DIR "$1/vein_mask_edt_comp"\
    DATALOADER.NUM_WORKERS 16\
    TEST.PATCHES_PER_ITER 64
    
