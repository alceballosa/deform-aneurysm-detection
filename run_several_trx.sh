#!/bin/bash


export id_gpu=0,1,2,3
export family="trx"

export model="trx_fa_input_edt"



./src/run_inference_cmha.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_cmha.sh $family $model "0039999" 0.8 $id_gpu

./src/run_inference_external.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_external.sh $family $model "0039999" 0.8 $id_gpu

./src/run_inference_test.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_test.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_test.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_test.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_test.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_test.sh $family $model "0039999" 0.8 $id_gpu
