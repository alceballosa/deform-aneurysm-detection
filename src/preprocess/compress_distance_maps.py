import os
import sys
from glob import glob

import edt
import numpy as np
import SimpleITK as sitk
import tqdm
from joblib import Parallel, delayed
from pathlib import Path


def compress_scan(path_file, path_target, threshold):

    scan = sitk.ReadImage(path_file)
    scan_name = Path(path_file).name
    array = sitk.GetArrayFromImage(scan)
    array[array < -threshold] = -threshold
    target_scan = sitk.GetImageFromArray(array)
    target_scan.CopyInformation(scan)
    path_target_scan = path_target + "/" + scan_name
    sitk.WriteImage(target_scan, path_target_scan, useCompression=True)


if __name__ == "__main__":
    edt_dir = sys.argv[1]
    threshold = int(sys.argv[2])
    target_dir = edt_dir + "_comp"
    try:
        threads = int(sys.argv[3])
    except IndexError:
        threads = 8
    os.makedirs(target_dir, exist_ok=True)
    edt_files = sorted(list(glob(f"{edt_dir}/*.nii.gz")))
    num_workers = threads if len(edt_files) > 8 else len(edt_files)
    executor = Parallel(
        n_jobs=num_workers, backend="multiprocessing", prefer="processes", verbose=1
    )
    do = delayed(compress_scan)
    tasks = (do(im_f, target_dir, threshold) for im_f in edt_files)
    executor(tasks)
