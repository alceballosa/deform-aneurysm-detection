#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta2

# define the folder containing the data
# for training/evaluation, the scans should be in a folder called "og"
# and the labels in a folder called "og_label"

# for testing on unnanotated data, please refer to the other pipeline file

# define the path to your data here 
export path_base="/scratch/ceballosarroyo.a/aneurysm/cta_datasets/hospital140"

export path_og="${path_base}/og"
export path_label_og="${path_base}/og_label"
export path_resampled=${path_og}_0.4 
export path_label_resampled=${path_label_og}_0.4
export path_vessel_seg="${path_base}/crop_0.4_vessel"
export path_crop="${path_base}/crop_0.4"
export path_label_crop="${path_base}/crop_0.4_label"
export path_edt="${path_base}/crop_0.4_vessel_edt_comp"
export path_annotations="${path_base}/annotations.csv"
export path_cvs_outputs="${path_base}/cvs_temp"
export path_cvs_masks="${path_base}/cvs_mask"
export path_cvs_bbox="${path_base}/cvs_bbox"
export path_cvs_outputs="${path_base}/cvs_temp"
export path_cvs_masks="${path_base}/cvs_mask"
export path_cvs_bbox="${path_base}/cvs_bbox"
export path_vein_masks="${path_base}/vein_mask_edt_comp"


#Resample scans to 0.4mm spacing and crop them
# python src/preprocess/resample_scans.py ${path_og} ${path_label_og}
# python src/preprocess/crop_scans.py ${path_resampled} ${path_crop}
# python src/preprocess/crop_scans.py ${path_label_resampled} ${path_label_crop}
# mkdir ${path_crop}_split
# python src/preprocess/split_files.py ${path_crop} ${path_crop}_split
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
# done
# # Compute distance maps
# python src/preprocess/compute_distance_maps.py ${path_vessel_seg} ${path_edt} 90 1

# # Obtain bbox csv from segmentation files 
# python src/preprocess/get_bbox_csv_with_vein_artery.py ${path_label_crop} ${path_vessel_seg} ${path_edt} ${path_annotations}

# Get cvs masks
# python src/cvs_mask/compute_cvs.py ${path_crop} ${path_vessel_seg} ${path_cvs_outputs} ${path_cvs_masks} ${path_cvs_bbox} 

# python src/preprocess/compute_distance_maps.py ${path_cvs_masks} ${path_cvs_masks}_edt_comp 90 1

mkdir ${path_vein_masks}
python src/preprocess/compute_vein_minus_csv_distance_map.py ${path_vessel_seg} ${path_cvs_masks}  ${path_vein_masks} 90 1