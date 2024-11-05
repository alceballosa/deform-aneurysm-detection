import sys
from pathlib import Path

import SimpleITK as sitk
import tqdm
from joblib import Parallel, delayed

crop_threshold = 425
crop_padding = 15


def crop_scan(path_file, path_target):
    scan = sitk.ReadImage(path_file)
    scan_name = path_file.name
    array = sitk.GetArrayFromImage(scan)
    if array.shape[0] > crop_threshold:
        crop_point = array.shape[0] - crop_threshold
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([0, 0, crop_point])
        cropped_scan = crop.Execute(scan)
    else:
        cropped_scan = scan
    path_cropped_scan = path_target / scan_name
    sitk.WriteImage(cropped_scan, path_cropped_scan)
    return cropped_scan


if __name__ == "__main__":

    path_files = Path(sys.argv[1])
    path_target = Path(sys.argv[2])
    path_target.mkdir(exist_ok=True, parents=True)
    files = sorted(list(path_files.glob("*")))
    num_workers = 8
    executor = Parallel(
        n_jobs=num_workers, backend="multiprocessing", prefer="processes", verbose=1
    )
    do = delayed(crop_scan)
    tasks = (do(im_f, path_target) for im_f in files)
    executor(tasks)
