export CUDA_VISIBLE_DEVICES=1;
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/train_net.py\
    --num-gpus 1\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:52513"\
    --resume\


# ss -lptn 'sport = :6006'
