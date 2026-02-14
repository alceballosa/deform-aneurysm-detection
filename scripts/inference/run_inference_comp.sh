#!/bin/bash
# Usage: ./run_inference.sh <dataset_name> <family> <model> <checkpoint> <threshold> <num_gpus> <num_workers> <patches_per_iter> <root>
export ID_PORT=$(($RANDOM+20000))
export PYTHONPATH=$(pwd):$PYTHONPATH

python src/train_net.py\
    --num-gpus $6\
    --config-file "./configs/$2/$3.yaml"\
    --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --eval-only\
    MODEL.WEIGHTS $4\
    DATA.DIR.ROOT "$9"\
    DATA.DIR.VAL.SCAN_DIR "$1/crop_0.4_compressed"\
    DATA.DIR.VAL.VESSEL_DIR "$1/crop_0.4_artery_edt_comp_32"\
    DATA.DIR.VAL.CVS_DIR "$1/vein_mask_edt_comp"\
    DATA.DIR.VAL.LABEL_DIR "$1/crop_0.4_label"\
    DATALOADER.NUM_WORKERS $7\
    TEST.PATCHES_PER_ITER $8
