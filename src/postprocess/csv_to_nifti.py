import pdb
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from detectron2.engine import default_argument_parser
from src.config import get_inference_iters_id, setup
from tqdm import tqdm

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args)
    checkpoint = get_inference_iters_id(cfg.POSTPROCESS.CHECKPOINT)
    path_outputs_csv = Path(cfg.OUTPUT_DIR) / f"inference_{checkpoint}/predict.csv"
    df_outputs = pd.read_csv(path_outputs_csv)
    path_outputs_nifti = Path(cfg.OUTPUT_DIR) / f"inference_{checkpoint}/nifti"
    path_outputs_nifti.mkdir(parents=True, exist_ok=True)
    threshold = float(cfg.POSTPROCESS.THRESHOLD)
    path_label_files = Path(cfg.DATA.DIR.VAL.SCAN_DIR)

    all_nifti_label_files = list(path_label_files.glob("*.nii.gz"))
    print(f"\nConverting outputs into nifti format for checkpoint {checkpoint}")
    for file in tqdm(all_nifti_label_files):
        nifti_label_header = sitk.ReadImage(str(file))
        # get spacing
        spacing = nifti_label_header.GetSpacing()
        # get shape of array in nifti_label_header
        shape_y, shape_x, shape_z = nifti_label_header.GetSize()
        pred_mask = np.zeros((shape_z, shape_x, shape_y)).astype(np.uint8)
        df_outputs_scan = df_outputs[df_outputs["seriesuid"] == file.name]
        df_outputs_scan = df_outputs_scan[df_outputs_scan["probability"] > threshold]
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
