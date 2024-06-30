path_src = "/home/ceballosarroyo.a/workspace/medical/cta-det2/src"
import sys
from pathlib import Path

import pandas as pd

sys.path.append(path_src)
import json
import sys

import numpy as np
import pandas as pd
from log_utils import setup_logger
from sklearn.metrics._ranking import _binary_clf_curve


def get_threshold(value, row):

    tr = 350
    if value > tr:  # and row_num in valids:
        threshold = value - tr
        # print(value, row)
    else:
        threshold = 0
    return threshold


def remove_by_size(df_preds, size_dict):
    df_preds = df_preds.copy()

    # determine for each row the threshold at which to remove pred
    df_preds["threshold"] = df_preds["seriesuid"].apply(
        lambda row: get_threshold(size_dict[row][2], row)
    )
    df_preds["threshold"] = df_preds["threshold"].astype(int)
    # if row is smaller than threshold, remove it
    df_preds = df_preds[df_preds["coordZ"] > df_preds["threshold"]]

    return df_preds


from froc import FROCEvaluator


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
    "val": Path(
        "/home/ceballosarroyo.a/workspace/medical/cta-det2/labels/internal_test_crop_0.4.csv"
    ),
    "ext": root / "labels/external_0.4_crop_vol.csv",
    "val_no_crop": "/work/vig/Datasets/aneurysm/test0.4.csv",
    "priv": root / "labels/hospital_0.4.csv",
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
    "priv": root / "labels/metadata/hospital_meta.json",
}

df_labels_val_with_sizes = pd.read_csv(root / "labels/test0.4_crop_vol.csv")

max_fppi = 10.0
min_fppi = 0.0
fp_scale = "linear"
fppi_thrs = [0.5, 1.0, 2.0, 4.0, 8.0]
n_bootstraps = 10000
iou_thr = 0.5


# exp_base = "deform_decoder_only_rec_16_heads"

exp_base = sys.argv[1]
thresholds = {
    "dense_bn_64_infer_PRIV": 0.79,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_PRIV": 0.972,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_1l": 0.95,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_1l_EXT": 0.99,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_2l": 0.9,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_2l_EXT": 0.9,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug": 0.95,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug_EXT": 0.95,
    "dense_bn_64_8_queries_crop": 0.9,
    "dense_bn_64_8_queries_crop_EXT": 0.92,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug": 0.8,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug_EXT": 0.8,
    "dense_bn_64_infer": 0.80,
    "dense_bn_64_infer_EXT": 0.81,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe": 0.90,
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT": 0.965,
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe": 0.95,
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe_EXT": 0.95,
    ".nndet_crop": 0.6,
    ".nndet_crop_EXT": 0.6,
    ".nndet_crop_PRIV": 0.9,
    "deform_decoder_only_non_rec_BEST_cropinf": 0.94,
    "deform_decoder_only_non_rec_BEST_cropinf_EXT": 0.95,
    "deform_decoder_only_non_rec_16_heads_random": 0.93,
    "deform_decoder_only_non_rec_16_heads_random_EXT": 0.93,
}
inf_appends = {
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_1l": "50k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_1l_EXT": "50k",
    "dense_bn_64_infer_PRIV": "hieu",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_PRIV": "66k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_2l": "60k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_2l_EXT": "60k",
    "dense_bn_64_8_queries_crop": "final_crop",
    "dense_bn_64_8_queries_crop_EXT": "final",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug": "50k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_noaug_EXT": "50k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe_EXT": "66k",
    "adeform_decoder_only_non_rec_crop_vessel_pe_gpe": "66k",
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe_EXT": "final",
    "adeform_decoder_only_non_rec_crop_vessel_start_gpe": "final",
    "dense_bn_64_infer_EXT": "hieu",
    "dense_bn_64_infer": "hieu",
    ".nndet_crop_EXT": "nndet",
    ".nndet_crop": "nndet",
    ".nndet_crop_PRIV": "nndet",
    "deform_decoder_only_non_rec_BEST_cropinf": "30k",
    "deform_decoder_only_non_rec_BEST_cropinf_EXT": "40k",
    "deform_decoder_only_non_rec_16_heads_random": "30k",
    "deform_decoder_only_non_rec_16_heads_random_EXT": "40k",
}

inf_append = inf_appends[exp_base]
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

out_dir = root / f"outputs/{exp}/iou{iou_thr:.1f}_froc_{inf_append}_crop"
is_crop_in_exp = "crop" in exp or mode in ["ext", "train", "priv"]

if is_crop_in_exp:

    # val_meta_path = root / "labels/internal_test_meta_crop.json"
    if "_TI" in exp:
        label_file = label_files["train"]
        val_meta_path = meta_files["train"]
    elif "_EXT" in exp:
        label_file = label_files["ext"]
        val_meta_path = meta_files["ext"]
    elif "_PRIV" in exp:
        label_file = label_files["priv"]
        val_meta_path = meta_files["priv"]
    else:  # VALIDATION - default
        label_file = label_files["val"]
        val_meta_path = meta_files["val"]

else:
    val_meta_path = meta_files["val_no_crop"]
    label_file = label_files["val_no_crop"]
total_volumes = {"val": 152, "ext": 138, "priv": 38}

size_files = {
    "train_no_crop": root / "labels/sizes/scan_sizes_train.json",
    "val_no_crop": root / "labels/sizes/scan_sizes_test.json",
}

df_labels = pd.read_csv(label_file)

# path_sizes = size_files["val_no_crop"]
# sizes = pd.read_json(path_sizes)
with open(val_meta_path, "r") as f:
    meta = json.load(f)
# print(len(preds))
# preds = preds[preds["probability"] > 0.9]
# print("After2",len(preds))

# print(meta)

vols = total_volumes[mode]
logger = setup_logger(output=out_dir, name=__name__ + str(iou_thr))

if "iom" in preds.columns:
    preds = preds[preds["iom"] > 0.1]
    print("aa")


with open(val_meta_path, "r") as f:
    meta = json.load(f)
evaluator = FROCEvaluator(
    label_file=label_file,
    preds=preds,
    logger=logger,
    iou_thr=iou_thr,
    out_dir=out_dir,
    max_fppi=max_fppi,
    fppi_thrs=fppi_thrs,
    min_fppi=min_fppi,
    n_bootstraps=n_bootstraps,
    n_workers=n_workers,
    fp_scale=fp_scale,
    meta_data=meta,
    use_world_xyz=not is_crop_in_exp,
    exp_name=exp + "_" + inf_append,
    mode=mode,
)
evaluator.evaluate()


total_correct = 0
total_correct_any = 0
total_fp = 0
total_fn = 0

if mode == "val":
    df_labels[["volume", "min_axis", "maj_axis"]] = df_labels_val_with_sizes[
        ["volume", "min_axis", "maj_axis"]
    ]

correct_per_case = []
fp_per_case = []
fn_per_case = []
gt_per_case = []
correct_dets = []


t = thresholds[exp_base]

case_names = []

for key in evaluator._match_results.keys():
    gt_per_case.append(len(df_labels[df_labels["seriesuid"] == key]))
    un_matched_gt = evaluator._match_results[key]["aneurysm"]["un_matched_gt"]
    all_un_matched_gt = evaluator._match_results[key]["aneurysm"]["all_un_matched_gt"]
    filtered_umgt = [x for x in all_un_matched_gt if x[0] > t]
    un_matched_gt = filtered_umgt[-1][-1] if len(filtered_umgt) > 0 else np.array([])
    case_names.append(key)
    if len(un_matched_gt) > 1:
        un_matched_gt = un_matched_gt[0:-1]
        total_correct_any += np.sum(un_matched_gt == 0)
        correct_dets += list(un_matched_gt == 0)
        fn = np.sum(un_matched_gt)
    else:
        fn = len(df_labels[df_labels["seriesuid"] == key])
        correct_dets += [0] * len(df_labels[df_labels["seriesuid"] == key])
    fn_per_case.append(fn)
    scores = evaluator._match_results[key]["aneurysm"]["scores"].copy()
    gt = evaluator._match_results[key]["aneurysm"]["gts"].copy()
    gt[scores < t] = 0
    correct_per_case.append(np.sum(gt))
    fp_per_case.append(np.sum(gt[scores >= t] == 0))
    # for res in evaluator._match_results[key]:

total_fp = np.sum(fp_per_case)
total_fn = np.sum(fn_per_case)
total_correct = np.sum(correct_per_case)
total_gt = np.sum(gt_per_case)

spacing = 0.4
volume_per_voxel = spacing**3
if "volume" not in df_labels.columns:
    df_labels["volume"] = 1
df_labels["volume_mm"] = df_labels["volume"] * volume_per_voxel
df_labels["diameter"] = 1.445 * (3 * df_labels["volume_mm"] / (4 * np.pi)) ** (1 / 3)
df_labels["size"] = df_labels["diameter"].apply(classify_sizes)
df_labels["partition"] = df_labels["seriesuid"].apply(filterfun)
df_labels["detected"] = np.array(correct_dets).astype(np.int8)


print("Value counts across sizes:", df_labels["size"].value_counts())

print(
    vols,
    total_gt,
    "TP",
    total_correct / total_gt,
    "FP",
    total_fp / vols,
    "Aneurysms",
    total_gt,
    f"{int(np.sum(fn_per_case))}/{total_gt}",
)


df_labels.to_csv("predictions_internal_best_model.csv", index=False)

print("Performance across sizes", df_labels.groupby("size")["detected"].mean())

healthy_case_indexes = np.where(np.array(gt_per_case) == 0)[0]


healthy_fp = np.array(fp_per_case)[healthy_case_indexes]
healthy_fp = healthy_fp[healthy_fp > 0]
print(
    "P-Specificifty",
    len(healthy_case_indexes) - len(healthy_fp),
    len(healthy_case_indexes),
)

sick_case_indics = np.where(np.array(gt_per_case) > 0)[0]
sick_fn = np.array(fn_per_case)[sick_case_indics]
print("P-Sensitivity", len(sick_case_indics) - sum(sick_fn > 0), len(sick_case_indics))
