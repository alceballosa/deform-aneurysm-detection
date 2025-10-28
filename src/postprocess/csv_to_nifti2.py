import pdb
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import sys 

if __name__ == "__main__":
    path_root = "/projects/vig/Datasets/aneurysm/cta_datasets"
    path_labels = {
        "internal_train": path_root + "/internal_train/crop_0.4_label",
        "internal_test": path_root + "/internal_test/crop_0.4_label",
        "cmha": path_root + "/cmha/crop_0.4_label",
        "external": path_root + "/external/crop_0.4_label",
        "hospital140": path_root + "/hospital140/mask_aneurysm/crop_0.4_label_aneurysm",
    }
    args = sys.argv
    dataset = args[1]
    thres = float(args[2])

    path_results = Path(f"./results/{dataset}")

    for path_model in path_results.glob("*"):
        for path_checkpoint in path_model.glob("*"):
            cp_name = path_checkpoint.name
            if "inference" in cp_name:
                file = path_checkpoint / "predict_roi_dilated.csv"
                # check if file exists 
                if not file.exists():
                    print(f"File {file} does not exist, skipping...")
                    continue
                path_outputs_nifti = path_checkpoint / "nifti"
                path_outputs_nifti.mkdir(parents=True, exist_ok=True)
                df_outputs = pd.read_csv(file)
                print(path_labels[dataset])
                path_label_files = Path(path_labels[dataset])
                all_nifti_label_files = list(path_label_files.glob("*.nii.gz"))
                print(f"\nConverting outputs into nifti format for checkpoint {path_model.name} {cp_name}")
                for file in tqdm(all_nifti_label_files):
                    path_output = str(path_outputs_nifti / file.name)
                    # check if exists 
                    if Path(path_output).exists():
                        print(f"File {path_output} already exists, skipping...")
                        continue
                    nifti_label_header = sitk.ReadImage(str(file))
                    # get spacing
                    spacing = nifti_label_header.GetSpacing()
                    # get shape of array in nifti_label_header
                    shape_y, shape_x, shape_z = nifti_label_header.GetSize()
                    pred_mask = np.zeros((shape_z, shape_x, shape_y)).astype(np.uint8)
                    df_outputs_scan = df_outputs[df_outputs["seriesuid"] == file.name]
                    df_outputs_scan = df_outputs_scan[df_outputs_scan["probability"] > thres]
                    for _, row in df_outputs_scan.iterrows():
                        x, y, z, probability = row[["coordX", "coordY", "coordZ", "probability"]]
                        d, h, w = row[["d", "h", "w"]]
                        sz, ez = int(z - d / 2), int(z + d / 2)
                        sy, ey = int(y - h / 2), int(y + h / 2)
                        sx, ex = int(x - w / 2), int(x + w / 2)
                        pred_mask[sz:ez, sy:ey, sx:ex] = int(np.floor(100 * probability))
                    new_header = sitk.GetImageFromArray(pred_mask)
                    new_header.CopyInformation(nifti_label_header)
                    sitk.WriteImage(new_header, str(path_outputs_nifti / file.name))

