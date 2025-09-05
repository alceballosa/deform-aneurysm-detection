#!/bin/bash

export family="deform"

export model="$1"
export num_gpu="$2"
export root_folder="/scratch/ceballosarroyo.a/aneurysm/cta_datasets"
export dataset_folder = "${root_folder}/internal_test"

./run_inference.sh $dataset_folder $family $model "0063999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0059999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0055999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0051999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0047999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0039999" 0.8 $num_gpu

export dataset_folder = "${root_folder}/cmha"

./run_inference.sh $dataset_folder $family $model "0063999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0059999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0055999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0051999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0047999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0039999" 0.8 $num_gpu

export root_folder="/scratch/ceballosarroyo.a/aneurysm/cta_datasets/external"

./run_inference.sh $dataset_folder $family $model "0063999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0059999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0055999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0051999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0047999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0039999" 0.8 $num_gpu



