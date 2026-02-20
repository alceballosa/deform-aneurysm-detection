import shutil
from pathlib import Path

import pydicom
import SimpleITK as sitk
import tqdm

path_scans = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_healthy/og")

# get all filenames without .nii.gz
filenames = path_scans.glob("*.nii.gz")
filenames_fix = []
for fil in filenames:
    filenames_fix.append(str(fil.name).replace(".nii.gz", ""))
path_og_data = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/rsna/gdcmconv")

# for each filename, read the dicom files and get the RescaleIntercept and RescaleSlope
stats_header = ["RescaleIntercept", "RescaleSlope", "Minimum", "Maximum", "Filename"]
stats = []
for fil in tqdm.tqdm(filenames_fix):
    filename = fil.split("_")[0]
    dicom_path = path_og_data / filename
    dicom_files = list(dicom_path.glob("*.dcm"))
    # read one dicom file

    
    for dicom_file in dicom_files:
        ds = pydicom.dcmread(str(dicom_file))
        slice_thickness = ds.SliceThickness
        rescale_slope = None 
        rescale_intercept = None 
        try:
            rescale_slope = ds.RescaleSlope
            rescale_intercept = ds.RescaleIntercept
            
            break
        except:
            continue 
    if rescale_slope is None or rescale_intercept is None:
        print(f"Could not find RescaleSlope or RescaleIntercept for {fil}, skipping.")
        stats.append([0, 1, slice_thickness, 0, 0, fil])
        continue
    else:
        path_og_file = path_scans / (fil + ".nii.gz")

        stats.append([rescale_intercept, rescale_slope, slice_thickness, 0, 0, fil])
    #print(stats)
    if rescale_slope == 10:
        print(f"Found case with slope 10 and nonzero intercept: {fil}, slope: {rescale_slope}, intercept: {rescale_intercept}")


path_og = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_healthy/og")
path_target = Path(
    "/scratch/ceballosarroyo.a/aneurysm/mm_datasets/cta_rsna_healthy/og_rescaled"
)

# # mkdirs

path_target.mkdir(parents=True, exist_ok=True)
#path_label_target.mkdir(parents=True, exist_ok=True)

for intercept, slope, spacing, minimum, maximum, fil in tqdm.tqdm(stats):
    header = sitk.ReadImage(str(path_og / (fil + ".nii.gz")))
    arr = sitk.GetArrayFromImage(header)
    if spacing >= 2:
        print(f"Found case with spacing {spacing}: {fil}")
        continue
    
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
    #