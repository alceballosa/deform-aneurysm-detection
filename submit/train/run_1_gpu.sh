export CUDA_VISIBLE_DEVICES=0,1,2,3;
export PYTHONPATH=$(pwd):$PYTHONPATH
python src/train_net.py\
    --num-gpus 4\
    --config-file "./configs/deform/$1.yaml"\
    --resume\

# ss -lptn 'sport = :6006'
