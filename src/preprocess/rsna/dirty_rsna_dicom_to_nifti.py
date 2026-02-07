from pathlib import Path
import sys
import os
import dicom2nifti
import subprocess



path_src = Path("/projects/vig/Datasets/aneurysm/competition/series/")
files_src = sorted(list(path_src.glob("*")))
path_tgt = Path("/projects/vig/Datasets/aneurysm/mm_datasets/rsna/og")
os.makedirs(path_tgt, exist_ok=True)

errors = []
for f in files_src:
    print(f)
    try:
        # check if file exists
        if (path_tgt / f.name).exists():
            # count number of nii.gz files inside
            files = list((path_tgt / f.name).glob("*.nii.gz"))
            if len(files) > 0:
                print(f"Exists, skipping {f.name}")
                continue
        path_new_file = path_tgt / f.name
        os.makedirs(path_new_file, exist_ok=True)
        dicom2nifti.convert_directory(str(f), str(path_tgt / f.name), reorient=True)
    except Exception as e:
        print("Error with ", f)
        errors.append(f)
        print(e)
        continue
    

print(errors)