path_src = "/home/ceballosarroyo.a/workspace/medical/cta-det2/src"
import sys
from pathlib import Path

import pandas as pd

sys.path.append(path_src)

import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from froc import FROCEvaluator
from log_utils import setup_logger
from sklearn.metrics._ranking import _binary_clf_curve
from tabulate import tabulate
from torch.multiprocessing import Process, set_start_method
from tqdm import tqdm


def filterfun(text):
    if "Ts" in text:
        return "Int"
    elif "ExtA" in text:
        return "ExtA"
    else:
        return "ExtB"


def classify_sizes(x):
    if x <= 3:
        return "small"
    elif x <= 7:
        return "medium"
    else:
        return "large"


root = Path("/home/ceballosarroyo.a/workspace/medical/cta-det2/")

label_files = {
    "train": root / "labels/train0.4_crop.csv",
    "val": root / "labels/gt/internal_test_crop_0.4.csv",
    "ext": root / "labels/gt/external_crop_0.4.csv",
    "val_no_crop": "/work/vig/Datasets/aneurysm/test0.4.csv",
    "hospital": root / "labels/gt/hospital_crop_0.4.csv",
}
size_files = {
    "train_no_crop": root / "labels/sizes/scan_sizes_train.json",
    "val_no_crop": root / "labels/sizes/scan_sizes_test.json",
}

meta_files = {
    "ext": root / "labels/metadata/external_crop_meta.json",
    "train": root / "labels/metadata/internal_train_crop_meta.json",
    "val": root / "labels/metadata/internal_test_meta_crop.json",
    "val_no_crop": root / "labels/metadata/internal_test_meta.json",
    "hospital": root / "labels/metadata/hospital_meta.json",
}

df_labels_val_with_sizes = pd.read_csv(root / "labels/test0.4_crop_vol.csv")

max_fppi = 10.0
min_fppi = 0.0
fp_scale = "linear"
fppi_thrs = [0.5, 1.0, 2.0, 4.0, 8.0]
n_bootstraps = 10000
iou_thr = 0.1


# exp_base = "dense_bn_64_infer_EXT"
# inf_append = "hieu"
exp_base = "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT"
inf_append = "66k"
# exp_base = ".nndet_crop_EXT"
# inf_append = "nndet"
exps = [exp_base, exp_base + "_TI", exp_base + "_EXT"]

exp = exps[0]
path_inf = "inference_" + inf_append
print(f"Running iou_thr: {iou_thr} at {inf_append}")
if "TI" in exp:
    mode = "train"
elif "EXT" in exp:
    mode = "ext"
elif "PRIV" in exp:
    mode = "priv"
else:
    mode = "val"

n_workers = 8
out_dir = root / f"outputs/{exp}/iou{iou_thr:.1f}_froc_{inf_append}"
path_preds = root / f"outputs/{exp}/{path_inf}/predict.csv"

preds = pd.read_csv(path_preds)
thresholds = {
    "dense_bn_64_infer": 0.80,
    "dense_bn_64_infer_EXT": 0.81,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe": 0.90,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT": 0.95,
    ".nndet_crop": 0.6,
    ".nndet_crop_EXT": 0.6,
    "deform_decoder_only_non_rec_BEST_cropinf": 0.94,
    "deform_decoder_only_non_rec_BEST_cropinf_EXT": 0.95,
}

t = thresholds[exp_base]
path_vessels = {
    "val": (
        Path("/work/vig/Datasets/aneurysm/internal_test/crop_0.4_vessel/"),
        Path("/work/vig/Datasets/aneurysm/internal_test/crop_0.4_vessel_v2/"),
    ),
    "ext": (
        Path("/work/vig/Datasets/aneurysm/external/crop_0.4_vessel/"),
        Path("/work/vig/Datasets/aneurysm/external/crop_0.4_vessel_v2/"),
    ),
}

vessel_cache = {}
114, 92, 220



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


# if main

preds["iom"] = preds.apply(
    lambda row: get_intersection_with_vessels(row, mode, i=0), axis=1
)
preds["iom_v2"] = preds.apply(
    lambda row: get_intersection_with_vessels(row, mode, i=1), axis=1
)
# add vessel IoM
preds.to_csv(str(root / f"outputs/{exp}/{path_inf}/predict.csv"))
