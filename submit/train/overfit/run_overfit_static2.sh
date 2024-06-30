export PYTHONPATH=$(pwd):$PYTHONPATH
python src/train_net.py\
    --num-gpus 1\
    --config-file "./configs/dense_parq/dense_bn_64_overfit_batch_static2.yaml"\