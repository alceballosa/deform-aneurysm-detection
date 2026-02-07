import os
import sys
from glob import glob

import edt
import numpy as np
import SimpleITK as sitk
import tqdm
from joblib import Parallel, delayed


def compress_background(file, target_dir, threshold=-512):
    scan_seriesuid = file.split("/")[-1]
    save_file_path = os.path.join(target_dir, scan_seriesuid)
    # check if exists
    if os.path.exists(save_file_path):
        return
    im_header = sitk.ReadImage(file)
    im = sitk.GetArrayFromImage(im_header)
    im[im <= threshold] = threshold

    im_comp_header = sitk.GetImageFromArray(im)
    im_comp_header.CopyInformation(im_header)
    
    sitk.WriteImage(im_comp_header, save_file_path)

if __name__ == "__main__":
    scan_dir = sys.argv[1]
    target_dir = sys.argv[2]
    try:
        threads = int(sys.argv[3])
    except IndexError:
        threads = 8

    os.makedirs(target_dir, exist_ok=True)
    scan_files = sorted(list(glob(f"{scan_dir}/*.nii.gz")))
    executor = Parallel(
        n_jobs=threads, backend="multiprocessing", prefer="processes", verbose=1
    )
    do = delayed(compress_background)
    tasks = (do(im_f, target_dir) for im_f in scan_files)
    executor(tasks)

    print("All compressed images computed!")