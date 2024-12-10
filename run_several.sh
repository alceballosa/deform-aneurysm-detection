#!/bin/bash


export id_gpu=0,1
export model="decoder_only_no_rec_input_edt_cvs_nc"

./src/run_inference_local.sh $model "final" 0.8 $id_gpu
./src/run_inference_local.sh $model "0059999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0055999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0053999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0051999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0049999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0047999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0039999" 0.8 $id_gpu
./src/run_inference_local.sh $model "0029999" 0.8 $id_gpu


