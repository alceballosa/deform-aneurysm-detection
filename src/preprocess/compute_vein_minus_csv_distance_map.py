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


def compute_distance_map(file, file_cvs, target_dir, compress):
    scan_seriesuid = file.split("/")[-1]
    save_file_path = os.path.join(target_dir, scan_seriesuid)
    # check if exists
    if os.path.exists(save_file_path):
        return
    im_header = sitk.ReadImage(file)
    im = sitk.GetArrayFromImage(im_header)
    im = im == 2 # only veins
    
    im_cvs_header = sitk.ReadImage(file_cvs)
    im_cvs = sitk.GetArrayFromImage(im_cvs_header)

    # take only areas that are not in the CVS

    im = im * (im_cvs == 0)
    


    if len(np.unique(im)) < 2:
        print("Potential error on ", file)
    im_dist = edt.sdf(im, black_border=False)#, parallel=1)
    if compress == 1:
        threshold = 128
        im_dist[im_dist < -threshold] = -threshold
    # print(im_dist.shape)
    im_dist_header = sitk.GetImageFromArray(im_dist)
    im_dist_header.CopyInformation(im_header)
    
    sitk.WriteImage(im_dist_header, save_file_path)

if __name__ == "__main__":
    vessel_dir = sys.argv[1]
    cvs_dir = sys.argv[2]
    target_dir = sys.argv[3]
    try:
        threads = int(sys.argv[4])
    except IndexError:
        threads = 8
    compress = sys.argv[5]
    try:
        compress = int(compress)
    except:
        compress = 0

    os.makedirs(target_dir, exist_ok=True)
    vessel_files = sorted(list(glob(f"{vessel_dir}/*.nii.gz")))
    cvs_files = sorted(list(glob(f"{cvs_dir}/*.nii.gz")))
    if len(vessel_files) != len(cvs_files):
        raise ValueError("Number of vessel and CVS files do not match")
    # pair the files
    vessel_files = sorted(vessel_files)
    cvs_files = sorted(cvs_files)
    files = list(zip(vessel_files, cvs_files))
    executor = Parallel(
        n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
    )
    do = delayed(compute_distance_map)
    tasks = (do(im_f, im_cvs, target_dir, compress) for (im_f, im_cvs) in files)
    executor(tasks)

    print("All distance maps computed!")