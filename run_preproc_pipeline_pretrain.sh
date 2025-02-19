#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 

# define the folder containing the data
# the scans should be in a folder called "og"

# for training or evaluating on annotated data, please refer to the other pipeline file

# define the path to your data here 
export path_base="/data/pt_sinoct"

export path_og="${path_base}/og"
export path_resampled=${path_og}_0.4 
export path_vessel_seg="${path_base}/crop_0.4_vessel"
export path_crop="${path_base}/crop_0.4"
export path_edt="${path_base}/crop_0.4_vessel_edt"
export path_cvs_outputs="${path_base}/cvs_temp"
export path_cvs_masks="${path_base}/cvs_mask"
export path_cvs_bbox="${path_base}/cvs_bbox"


# Resample scans to 0.4mm spacing and crop them
# python src/preprocess/resample_scans.py ${path_og} 
# python src/preprocess/crop_scans.py ${path_resampled} ${path_crop}

# Split all files in the folder into subfolders with 20 files each using bash

# python src/preprocess/split_files.py ${path_crop} ${path_crop}_split

# # run a for loop over every subfolder in the split folder

# mkdir ${path_vessel_seg}

# for folder in ${path_crop}_split/*; do
#     if [ -d "$folder" ]; then
#         # Run vessel segmentation
#         mkdir ${folder}_temp
#         sudo docker run --gpus all -it --rm -v ${folder}_temp/:/Data/aneurysmDetection/output_path/  -v ${folder}/:/Data/aneurysmDetection/input_cta/ --shm-size=24g --ulimit memlock=-1 vessel_seg:latest python /Work/scripts/extractVessels.py -d /Data/aneurysmDetection/input_cta/ /Data/aneurysmDetection/output_path -m 'Prediction' -t 16 -s 0.5 -g 0 --continue_prediction
#         # Keep only relevant files 
        
#         sudo rm  ${folder}_temp/Predictions/CA_*
#         sudo rm ${folder}_temp/Predictions/*.json 
#         cp ${folder}_temp/Predictions/* ${path_vessel_seg}/
#         sudo rm -rf ${folder}_temp
#     fi
#     # Run vessel segmentation
#     # mkdir ${folder}_temp
#     # sudo docker run --gpus all -it --rm -v ${folder}_temp/:/Data/aneurysmDetection/output_path/  -v ${folder}/:/Data/aneurysmDetection/input_cta/ --shm-size=24g --ulimit memlock=-1 vessel_seg:latest python /Work/scripts/extractVessels.py -d /Data/aneurysmDetection/input_cta/ /Data/aneurysmDetection/output_path -m 'Prediction' -t 16 -s 0.5 -g 1
#     # # Keep only relevant files 
    
#     # sudo rm  ${folder}_temp/Predictions/CA_*
#     # sudo rm ${folder}_temp/Predictions/*.json 
#     # cp ${folder}_temp/Predictions/* ${path_vessel_seg}/
#     # sudo rm -rf ${folder}_temp
# done

# # # Compute distance maps
python src/preprocess/compute_distance_maps.py ${path_vessel_seg} ${path_edt} 90

# # # # Get cvs masks
# python src/cvs_mask/compute_cvs.py ${path_crop} ${path_vessel_seg} ${path_cvs_outputs} ${path_cvs_masks} ${path_cvs_bbox} 

# # python src/preprocess/compute_distance_maps.py ${path_cvs_masks} ${path_cvs_masks}_edt

# # # python src/preprocess/compress_distance_maps.py ${path_cvs_masks}_edt 128 90
# # python src/preprocess/compress_distance_maps.py ${path_edt} 128 90
