#!/bin/bash


export id_gpu=0,1,2,3
export family="trx"

export model="cnn_4l_input_edt"


./src/run_inference_priv.sh $family $model "0047999" 0.8 $id_gpu


export model="cnn_4l_input_edt_cvs"


./src/run_inference_priv.sh $family $model "final" 0.8 $id_gpu

