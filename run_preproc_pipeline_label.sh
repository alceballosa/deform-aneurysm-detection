#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate cta 

# define the folder containing the data
# for training/evaluation, the scans should be in a folder called "og"
# and the labels in a folder called "og_label"

# for testing on unnanotated data, please refer to the other pipeline file

# define the path to your data here 
export path_base="/projects/vig/Datasets/aneurysm/cta_datasets/hospital140"

export path_og="${path_base}/og"
export path_label_og="${path_base}/og_label"
export path_resampled=${path_og}_0.4 
export path_label_resampled=${path_label_og}_0.4
export path_vessel_seg="${path_base}/crop_0.4_vessel"
export path_crop="${path_base}/crop_0.4"
export path_label_crop="${path_base}/crop_0.4_label"
export path_edt="${path_base}/crop_0.4_vessel_edt"
export path_annotations="${path_base}/annotations.csv"
export path_cvs_outputs="${path_base}/cvs_temp"
export path_cvs_masks="${path_base}/cvs_mask"
export path_cvs_bbox="${path_base}/cvs_bbox"


# Resample scans to 0.4mm spacing and crop them
python src/preprocess/resample_masks.py ${path_label_og}
python src/preprocess/crop_scans.py ${path_label_resampled} ${path_label_crop}

# get annotations
python src/preprocess/get_bbox_csv.py ${path_label_crop}  ${path_annotations}
