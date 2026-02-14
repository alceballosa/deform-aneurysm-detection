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
from joblib import Parallel, delayed


def compute_distance_map(file, target_dir, compress, type_e):
    scan_seriesuid = file.split("/")[-1]
    save_file_path = os.path.join(target_dir, scan_seriesuid)
    # check if exists
    if os.path.exists(save_file_path):
        return
    im_header = sitk.ReadImage(file)
    im = sitk.GetArrayFromImage(im_header)
    if type_e == "artery":
        im = im == 1 # only arteries
    elif type_e == "vein":
        im = im == 2 # only veins
    else:
        raise ValueError(f"Unknown type_e: {type_e}")
    im = im.astype(int)
    if len(np.unique(im)) < 2:
        print("Potential error on ", file)
    im_dist = edt.sdf(im, black_border=False)#, parallel=1)
    if compress == 1:
        threshold = 32
        im_dist[im_dist < -threshold] = -threshold
    # print(im_dist.shape)
    im_dist_header = sitk.GetImageFromArray(im_dist)
    im_dist_header.CopyInformation(im_header)
    
    sitk.WriteImage(im_dist_header, save_file_path)

if __name__ == "__main__":
    vessel_dir = sys.argv[1]
    target_dir = sys.argv[2]
    try:
        threads = int(sys.argv[3])
    except IndexError:
        threads = 8
    compress = sys.argv[4]
    try:
        compress = int(compress)
    except:
        compress = 0
    
    type_e = sys.argv[5]

    os.makedirs(target_dir, exist_ok=True)
    vessel_files = sorted(list(glob(f"{vessel_dir}/*.nii.gz")))
    executor = Parallel(
        n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
    )
    do = delayed(compute_distance_map)
    tasks = (do(im_f, target_dir, compress, type_e) for im_f in vessel_files)
    executor(tasks)

    print("All distance maps computed!")