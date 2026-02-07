import shutil
from pathlib import Path

import pydicom
import SimpleITK as sitk
import tqdm

path_scans = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_ane/og_0.4")

# get all filenames without .nii.gz
filenames = path_scans.glob("*.nii.gz")
filenames_fix = []
for fil in filenames:
    filenames_fix.append(str(fil.name).replace(".nii.gz", ""))
path_og_data = Path("/projects/vig/Datasets/aneurysm/competition/series")

# for each filename, read the dicom files and get the RescaleIntercept and RescaleSlope
stats_header = ["RescaleIntercept", "RescaleSlope", "Minimum", "Maximum", "Filename"]
stats = []
for fil in tqdm.tqdm(filenames_fix):
    filename = fil.split("_")[0]
    dicom_path = path_og_data / filename
    dicom_files = list(dicom_path.glob("*.dcm"))
    # read one dicom file
    ds = pydicom.dcmread(str(dicom_files[20]))
    try:
        rescale_intercept = ds.RescaleIntercept
        rescale_slope = ds.RescaleSlope

        if rescale_slope == 1 and rescale_intercept == 0:
            continue
        if rescale_slope != 10:
            continue

        path_og_file = path_scans / (fil + ".nii.gz")

        stats.append([rescale_intercept, rescale_slope, 0, 0, fil])
    except:
        # stats.append([fil, "N/A", "N/A"])
        continue

path_og = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_ane/og")
path_target = Path(
    "/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_ane/rescale3/og"
)
path_label_og = Path(
    "/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_ane/og_label"
)
path_label_target = Path(
    "/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_ane/rescale3/og_label"
)

# mkdirs

path_target.mkdir(parents=True, exist_ok=True)
path_label_target.mkdir(parents=True, exist_ok=True)

for intercept, slope, minimum, maximum, fil in tqdm.tqdm(stats):
    header = sitk.ReadImage(str(path_og / (fil + ".nii.gz")))
    arr = sitk.GetArrayFromImage(header)

    
    if slope == 10:
        arr = arr / 10
        arr += 1024 
    else:

        if intercept != 0:
            arr = arr - intercept

        if slope != 1:
            arr = arr / slope


    image_fixed = sitk.GetImageFromArray(arr)
    image_fixed.CopyInformation(header)
    sitk.WriteImage(image_fixed, str(path_target / (fil + ".nii.gz")))
    # copy label file
    shutil.copy2(
        path_label_og / (fil + ".nii.gz"), path_label_target / (fil + ".nii.gz")
    )
