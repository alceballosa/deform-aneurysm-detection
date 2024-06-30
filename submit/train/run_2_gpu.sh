export CUDA_VISIBLE_DEVICES=0,1;
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/dense_parq/$1.yaml"\
        --dist-url "tcp://127.0.0.1:52513"\
    --resume\

# ss -lptn 'sport = :6006'
