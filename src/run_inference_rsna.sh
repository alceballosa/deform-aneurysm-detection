#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta2 
export ID_PORT=$(($RANDOM+20000))
#cd /workspace/deform-aneurysm-detection
export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=$5 python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/$1/$2.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $3\
    DATA.DIR.ROOT "/scratch/ceballosarroyo.a/aneurysm/mm_datasets"\
    DATA.DIR.VAL.SCAN_DIR "cta_rsna_ane/crop_0.4"\
    DATA.DIR.VAL.LABEL_DIR "cta_rsna_ane/crop_0.4_label"\
    DATA.DIR.VAL.ANNOTATION_FILE "./labels/gt/internal_test_crop_0.4.csv"\
    DATA.DIR.VAL.VESSEL_DIR "cta_rsna_ane/crop_0.4_vessel_edt_comp"\
    DATA.DIR.VAL.CVS_DIR "cta_rsna_ane/cvs_mask_edt_comp"\
    DATALOADER.NUM_WORKERS 32\
    TEST.PATCHES_PER_ITER 128
    
