export CUDA_VISIBLE_DEVICES=0,1,2,3;
export PYTHONPATH=$(pwd):$PYTHONPATH
export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus 4\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:$ID_PORT"\
    --resume\

# ss -lptn 'sport = :6006'
