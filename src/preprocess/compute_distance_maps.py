"""
Script to compute distance maps from binary vessel masks. Expects ordered params:

1. Path to the directory containing binary vessel masks
2. Path to the directory to save the distance maps
3. Threshold to binarize the vessel masks
4. Number of threads to use for computation (optional, default=8)
"""

import os
import sys
from glob import glob

import edt
import numpy as np
import SimpleITK as sitk
import tqdm

if __name__ == "__main__":
    vessel_dir = sys.argv[1]
    target_dir = sys.argv[2]
    try:
        threads = int(sys.argv[3])
    except IndexError:
        threads = 8
    os.makedirs(target_dir, exist_ok=True)
    vessel_files = sorted(list(glob(f"{vessel_dir}/*.nii.gz")))
    for file in tqdm.tqdm(vessel_files):
        scan_seriesuid = file.split("/")[-1]
        save_file_path = os.path.join(target_dir, scan_seriesuid)
        # check if exists
        if os.path.exists(save_file_path):
            continue
        im_header = sitk.ReadImage(file)
        im = sitk.GetArrayFromImage(im_header)
        # im = (im > threshold).astype(np.uint8)
        im_dist = edt.sdf(im, black_border=False, parallel=threads)
        im_dist_header = sitk.GetImageFromArray(im_dist)
        im_dist_header.CopyInformation(im_header)
        
        sitk.WriteImage(im_dist_header, save_file_path)
    print("All distance maps computed!")
