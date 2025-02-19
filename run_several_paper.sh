#!/bin/bash


export id_gpu=2,3
export family="deform"

export model="decoder_only_no_rec_pe_edt_compressed"

# ./src/run_inference_local_2.sh $family $model "0039999" 0.8 $id_gpu

# export model="decoder_only_no_rec_pe_edt_compressed_EXT"

# ./src/run_inference_local_2.sh $family $model "0039999" 0.8 $id_gpu

# export model="decoder_only_no_rec_pe_edt_compressed_CMHA"

# ./src/run_inference_local_2.sh $family $model "0039999" 0.8 $id_gpu

export model="decoder_only_no_rec_pe_edt_compressed_PRIV"

./src/run_inference_local_2.sh $family $model "0039999" 0.8 $id_gpu

