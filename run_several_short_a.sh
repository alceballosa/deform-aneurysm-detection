#!/bin/bash


export id_gpu=0,1
export family="vivit"

export model="pt_vivit_1l_2ch_s_decoder_only_no_rec"


# ./src/run_inference_local_2.sh $family $model "0029999" 0.8 $id_gpu
# ./src/run_inference_local_2.sh $family $model "0027999" 0.8 $id_gpu
# ./src/run_inference_local_2.sh $family $model "0025999" 0.8 $id_gpu
# ./src/run_inference_local_2.sh $family $model "0023999" 0.8 $id_gpu
# ./src/run_inference_local_2.sh $family $model "0021999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0019999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0017999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0015999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0013999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0011999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0009999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0007999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0005999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0003999" 0.8 $id_gpu
./src/run_inference_local_2.sh $family $model "0001999" 0.8 $id_gpu


