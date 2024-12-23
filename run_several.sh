#!/bin/bash


export id_gpu=0,1,2,3
export family="vivit"
export model="vivit_decoder_only_no_rec_input_edt"

./src/run_inference_local.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0053999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0049999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0045999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0043999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0039999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0029999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0019999" 0.8 $id_gpu
./src/run_inference_local.sh $family $model "0009999" 0.8 $id_gpu
