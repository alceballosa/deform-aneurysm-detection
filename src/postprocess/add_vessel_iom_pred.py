import os
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from detectron2.engine import default_argument_parser
from src.config import get_inference_iters_id, setup
from tqdm import tqdm


def get_intersection_with_vessels(row, mode, i=0):
    case_name = row["seriesuid"]
    coordX, coordY, coordZ, d, h, w, p = (
        row["coordX"],
        row["coordY"],
        row["coordZ"],
        row["d"],
        row["h"],
        row["w"],
        row["probability"],
    )
    t = 0.5
    if p < t:
        # print(case_name, coordX, coordY, coordZ, d, h, w, p, 0)
        return "Low conf"
    file_vessels = path_vessels[mode][i] / f"{case_name}"
    print(file_vessels)
    header_vessels = sitk.ReadImage(str(file_vessels))
    array_vessels = sitk.GetArrayFromImage(header_vessels)
    row_array = np.zeros_like(array_vessels)
    array_vessels = array_vessels.copy()
    array_vessels = (array_vessels == 1).astype(np.uint8)
    row_array[
        max(0, int(coordZ - d // 2)) : min(row_array.shape[0], int(coordZ + d // 2)),
        max(0, int(coordY - h // 2)) : min(row_array.shape[1], int(coordY + h // 2)),
        max(0, int(coordX - w // 2)) : min(row_array.shape[2], int(coordX + w // 2)),
    ] = 1
    area_intersection = np.sum(np.logical_and(row_array, array_vessels))
    area_aneurysm = np.sum(row_array)
    int_over_min = area_intersection / area_aneurysm
    print(case_name, coordX, coordY, coordZ, d, h, w, p, int_over_min)
    return int_over_min


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
    path_label_files = str(cfg.DATA.DIR.VAL.SCAN_DIR)
    path_vessels = path_label_files + "_vessel_v2"
    all_path_vessel_files = os.listdir(path_vessels)

    artery_iom = []
    vein_iom = []
    for file in tqdm(all_path_vessel_files):
        file = Path(path_vessels)/file
        df_outputs_scan = df_outputs[df_outputs["seriesuid"] == file.name]
        # df_outputs_scan = df_outputs_scan[df_outputs_scan["probability"] > threshold]
        header_vessels = sitk.ReadImage(str(file))
        array_vessels_orig = sitk.GetArrayFromImage(header_vessels)
        for _, row in df_outputs_scan.iterrows():
            coordX, coordY, coordZ, d, h, w, p = (
                row["coordX"],
                row["coordY"],
                row["coordZ"],
                row["d"],
                row["h"],
                row["w"],
                row["probability"],
            )
            row_array = np.zeros_like(array_vessels_orig)
            array_vessels = array_vessels_orig.copy()

            # artery
            array_vessels = (array_vessels == 1).astype(np.uint8)
            row_array[
                max(0, int(coordZ - d // 2)) : min(
                    row_array.shape[0], int(coordZ + d // 2)
                ),
                max(0, int(coordY - h // 2)) : min(
                    row_array.shape[1], int(coordY + h // 2)
                ),
                max(0, int(coordX - w // 2)) : min(
                    row_array.shape[2], int(coordX + w // 2)
                ),
            ] = 1
            area_intersection = np.sum(np.logical_and(row_array, array_vessels))
            area_aneurysm = np.sum(row_array)
            int_over_min = area_intersection / area_aneurysm
            artery_iom.append(int_over_min)

            # vein
            row_array = np.zeros_like(array_vessels_orig)
            array_vessels = array_vessels_orig.copy()
            array_vessels = (array_vessels == 2).astype(np.uint8)
            row_array[
                max(0, int(coordZ - d // 2)) : min(
                    row_array.shape[0], int(coordZ + d // 2)
                ),
                max(0, int(coordY - h // 2)) : min(
                    row_array.shape[1], int(coordY + h // 2)
                ),
                max(0, int(coordX - w // 2)) : min(
                    row_array.shape[2], int(coordX + w // 2)
                ),
            ] = 1
            area_intersection = np.sum(np.logical_and(row_array, array_vessels))
            area_aneurysm = np.sum(row_array)
            int_over_min = area_intersection / area_aneurysm
            vein_iom.append(int_over_min)
    df_outputs["artery_iom"] = artery_iom
    df_outputs["vein_iom"] = vein_iom
    df_outputs.to_csv(path_outputs_csv, index=False)
