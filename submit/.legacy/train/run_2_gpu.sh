export PYTHONPATH=$(pwd):$PYTHONPATH
export ID_PORT=$(($RANDOM+20010))
python src/train_net.py\
    --num-gpus 2\
    --config-file "./configs/$1/$2.yaml"\
        --dist-url "tcp://127.0.0.1:${ID_PORT}"\
    --resume\

# ss -lptn 'sport = :6006'
