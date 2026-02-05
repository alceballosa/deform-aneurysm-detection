export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
export PYTHONPATH=$(pwd):$PYTHONPATH
export ID_PORT=$(($RANDOM+20000))
python src/train_net.py\
    --num-gpus 8\
    --config-file "./configs/deform/$1.yaml"\
        --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume\

# ss -lptn 'sport = :6006'
