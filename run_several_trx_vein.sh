#!/bin/bash


export id_gpu=0,1
export family="trx"

export model="cnn_4l_input_edt_vein_cvs_ogopt_clsw"

./src/run_inference_test_vein.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_test_vein.sh $family $model "0039999" 0.8 $id_gpu

./src/run_inference_cmha_vein.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_cmha_vein.sh $family $model "0039999" 0.8 $id_gpu

./src/run_inference_ext_vein.sh $family $model "final" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0063999" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0059999" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0055999" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0051999" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0047999" 0.8 $id_gpu
./src/run_inference_ext_vein.sh $family $model "0039999" 0.8 $id_gpu


