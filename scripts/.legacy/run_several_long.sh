#!/bin/bash


export id_gpu=0,1,2,3
export family="vivit"

export model="vivit_1l_fa_decoder_only_no_rec_input_edt_long"
./src/run_inference_local.sh $family $model "0089999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0079999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0069999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0095999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0093999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0091999" 0.8 $id_gpu

./src/run_inference_local.sh $family $model "0085999" 0.8 $id_gpu
