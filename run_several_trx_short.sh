#!/bin/bash

export family="$1"

export model="$2"
export num_gpu="$3"
#export root_folder="/projects/vig/Datasets/aneurysm/cta_datasets/"
export root_folder=""
export dataset="hospital140"
export dataset_folder="${dataset}"

#./run_inference.sh $dataset_folder $family $model "0003999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0007999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0009999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0015999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0013999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0012999" 0.8 $num_gpu




export dataset="cmha"
export dataset_folder="${root_folder}/${dataset}"
#mkdir /dev/shm/${dataset}
#cp -r $dataset_folder/crop_0.4 /dev/shm/${dataset}
# cp -r $dataset_folder/crop_0.4_vessel_edt_comp /dev/shm/${dataset}
#cp -r $dataset_folder/vein_mask_edt_comp /dev/shm/${dataset}
#export dataset_folder="/dev/shm/${dataset}"

./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0015999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0013999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0011999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0009999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0007999" 0.8 $num_gpu
#./run_inference.sh $dataset_folder $family $model "0003999" 0.8 $num_gpu

export dataset="internal_test"
export dataset_folder="${root_folder}/${dataset}"
#mkdir /dev/shm/${dataset}
#cp -r $dataset_folder/crop_0.4 /dev/shm/${dataset}
#cp -r $dataset_folder/crop_0.4_vessel_edt_comp /dev/shm/${dataset}
#cp -r $dataset_folder/vein_mask_edt_comp /dev/shm/${dataset}
#export dataset_folder="/dev/shm/${dataset}"


./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0015999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0013999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0011999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0009999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0007999" 0.8 $num_gpu
#./run_inference.sh $dataset_folder $family $model "0003999" 0.8 $num_gpu



export dataset="external"
export dataset_folder="${root_folder}/${dataset}"
#mkdir /dev/shm/${dataset}
#cp -r $dataset_folder/crop_0.4 /dev/shm/${dataset}
#cp -r $dataset_folder/crop_0.4_vessel_edt_comp /dev/shm/${dataset}
#cp -r $dataset_folder/vein_mask_edt_comp /dev/shm/${dataset}
#export dataset_folder="/dev/shm/${dataset}"


./run_inference.sh $dataset_folder $family $model "final" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0015999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0013999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0011999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0009999" 0.8 $num_gpu
./run_inference.sh $dataset_folder $family $model "0007999" 0.8 $num_gpu
#./run_inference.sh $dataset_folder $family $model "0003999" 0.8 $num_gpu


