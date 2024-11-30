export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3 python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:52513"\
    --resume\

# ss -lptn 'sport = :6006'
