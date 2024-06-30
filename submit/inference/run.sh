export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python src/train_net.py\
    --num-gpus 1\
    --config-file "./configs/deform/$1.yaml"\
    --eval-only\
    MODEL.WEIGHTS ./outputs/$1/model_$2.pth
