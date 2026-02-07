from enum import unique
from pathlib import Path
import sys
import os
import dicom2nifti
import subprocess



path_src = Path("/projects/vig/Datasets/aneurysm/cta_datasets/cmha_review/dicom")
files_src = sorted(list(path_src.glob("*")))
path_tgt = Path("/projects/vig/Datasets/aneurysm/cta_datasets/cmha_review/nifti")
os.makedirs(path_tgt, exist_ok=True)

errors = []
for f in files_src:

    # check if file exists
    if (path_tgt / f.name).exists():
        # count number of nii.gz files inside
        files = list((path_tgt / f.name).glob("*.nii.gz"))
        if len(files) > 0:
            print(f"Exists, skipping {f.name}")
            continue
    path_new_file = path_tgt / f.name
    os.makedirs(path_new_file, exist_ok=True)
    path_dicoms = f / "CTA images"
    path_dicoms_unified = f/ "CTA images unified"
    for dicom_file in path_dicoms.glob("*.DCM"):
        # read the series instance uid
        import pydicom
        ds = pydicom.dcmread(str(dicom_file))
        #print(dicom_file)
        # Acquisition Number 
        # make acquisition number = 1
        ds.AcquisitionNumber = 1
        ds.AcquisitionTime = "000000.000000"
        ds.IrradiationEventUID = "1.2.840.113619.2.334.3.3705602316.3.1417047048.530"
        # set trigger time
        ds.TriggerTime = 0
        # save to new file  
        # save to unified 
        os.makedirs(path_dicoms_unified, exist_ok=True)
        ds.save_as(str(path_dicoms_unified / dicom_file.name))
        
    dicom2nifti.convert_directory(str(path_dicoms_unified), str(path_new_file), reorient=True)
    #dicom2nifti.convert_directory(str(f/"CTA images"), str(path_tgt / f.name), reorient=True)



print(errors)