import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure
from utils.general import is_notebook

if is_notebook():
    root = Path("../../")
else:
    root = Path("./")

label_files = {
    "train": root / "labels/train0.4_crop.csv",
    "val": root / "labels/gt/internal_test_crop_0.4.csv",
    "ext": root / "labels/gt/external_crop_0.4.csv",
    "val_no_crop": "/work/vig/Datasets/aneurysm/test0.4.csv",
    "hospital": root / "labels/gt/hospital.csv",
    #"hospital": root / "labels/gt/hospital_crop_0.4_subsample.csv",
}
size_files = {
    "train_no_crop": root / "labels/sizes/scan_sizes_train.json",
    "val_no_crop": root / "labels/sizes/scan_sizes_test.json",
    "hospital": root / "labels/sizes/hospital.json",
}

meta_files = {
    "ext": root / "labels/metadata/external_crop_meta.json",
    "train": root / "labels/metadata/internal_train_crop_meta.json",
    "val": root / "labels/metadata/internal_test_meta_crop.json",
    "val_no_crop": root / "labels/metadata/internal_test_meta.json",
    "hospital": root / "labels/metadata/hospital_meta.json",
}


def get_threshold(value, row):

    tr = 425
    if value > tr:  # and row_num in valids:
        threshold = value - tr
        # print(value, row)
    else:
        threshold = 0
    return threshold


def classify_sizes(x):
    if x <= 3:
        return "small"
    elif x <= 7:
        return "medium"
    else:
        return "large"


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


def get_aneurysm_diameter(volume, spacing=None):
    # V = 4/3 * π * r^3
    if spacing:
        volume_per_voxel = spacing**3
        volume = volume * volume_per_voxel
    r = 1.445 * (3 * volume / (4 * np.pi)) ** (1 / 3)
    return r * 2


def get_gt_case_props(path_file):
    header = sitk.ReadImage(str(path_file))
    label = sitk.GetArrayFromImage(header)
    label = label.astype(np.uint8)
    all_labels = measure.label(label, background=0)
    props = measure.regionprops(all_labels)
    props = [
        {"bbox": prop.bbox, "centroid": prop.centroid, "area": prop.area}
        for prop in props
    ]

    return props


def process_label_files_centr(
    files_glia, files_label, path_props_cache, path_props_label_cache
):

    if path_props_cache.exists():
        with open(path_props_cache, "rb") as f:
            pred_props = pickle.load(f)
    else:
        pred_props = {}
        for i in range(0, len(files_glia)):

            case_name = files_glia[i].name
            case_props = get_gt_case_props(files_glia[i])
            pred_props[case_name] = case_props

        # save gt props
        with open(path_props_cache, "wb") as handle:
            pickle.dump(pred_props, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if path_props_label_cache.exists():
        with open(path_props_label_cache, "rb") as f:
            label_props = pickle.load(f)
    else:
        label_props = {}
        for i in range(0, len(files_label)):
            case_name = files_label[i].name
            case_props = get_gt_case_props(files_label[i])
            label_props[case_name] = case_props

        # save gt props
        with open(path_props_label_cache, "wb") as handle:
            pickle.dump(label_props, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pred_props, label_props


def compute_metrics_centroid_det_based(files, df_gt, df_pred, radius_factor=1):
    cols = ["tp", "fp", "fn"]
    rows = []
    aneurysms_found = []
    for i, _ in enumerate(files):

        case_name = files[i].name
        df_case = df_gt[df_gt["seriesuid"] == case_name]
        df_case_pred = df_pred[df_pred["seriesuid"] == case_name]

        if len(df_case) == 0:
            new_row = [0, len(df_case_pred), 0]
            rows.append(new_row)
        else:
            matches_scan = []
            # seriesuid,coordX,coordY,coordZ,w,h,d,lesion
            for aneurysm in df_case.iterrows():
                aneurysm = aneurysm[1]
                gt_x, gt_y, gt_z = (
                    aneurysm["coordX"],
                    aneurysm["coordY"],
                    aneurysm["coordZ"],
                )
                gt_w, gt_h, gt_d = aneurysm["w"], aneurysm["h"], aneurysm["d"]
                center_gt = np.array([gt_x, gt_y, gt_z])
                radius_gt = max(gt_w / 2, gt_h / 2, gt_d / 2) * radius_factor
                matches_aneurysm = []
                for pred in df_case_pred.iterrows():
                    pred = pred[1]
                    pred_w, pred_h, pred_d = pred["w"], pred["h"], pred["d"]
                    pred_x, pred_y, pred_z = (
                        pred["coordX"],
                        pred["coordY"],
                        pred["coordZ"],
                    )
                    center_pred = np.array(
                        [pred_x, pred_y, pred_z]
                    )  # * spacing + origin
                    distance_gt = np.linalg.norm(
                        np.array(center_gt) - np.array(center_pred)
                    )
                    radius = (
                        np.max((pred_w / 2, pred_h / 2, pred_d / 2)) * radius_factor
                    )
                    if distance_gt < radius_gt + radius:
                        matches_aneurysm.append(True)
                    else:
                        matches_aneurysm.append(False)

                matches_scan.append(matches_aneurysm)
            matches_scan = np.array(matches_scan).astype(int)
            true_positive_count = np.sum(matches_scan, axis=1)
            detected_aneurysms_scan = list(true_positive_count)
            aneurysms_found += detected_aneurysms_scan
            true_positive_count = np.sum(true_positive_count > 0)
            false_positive_count = np.sum(matches_scan, axis=0)
            false_positive_count = np.sum(false_positive_count == 0)
            false_negative_count = len(df_case) - true_positive_count
            new_row = [true_positive_count, false_positive_count, false_negative_count]
            # print(new_row)
            rows.append(
                [true_positive_count, false_positive_count, false_negative_count]
            )
    df_results = pd.DataFrame(rows, columns=cols)
    return df_results, aneurysms_found


def compute_metrics_glia(
    files: list, df_gt: pd.DataFrame, dict_gt_props: dict, metadata
):
    cols = ["tp", "fp", "fn"]
    rows = []
    aneurysms_found = []
    for i, _ in enumerate(files):

        case_name = files[i].name
        df_case = df_gt[df_gt["seriesuid"] == case_name]
        gt_props = dict_gt_props[case_name]

        if len(df_case) == 0:
            new_row = [0, len(gt_props), 0]
            rows.append([0, len(gt_props), 0])

        else:
            matches_scan = []
            aneurysms = df_case.values
            # seriesuid,coordX,coordY,coordZ,w,h,d,lesion
            for aneurysm in df_case.iterrows():
                aneurysm = aneurysm[1]
                gt_x, gt_y, gt_z = (
                    aneurysm["coordX"],
                    aneurysm["coordY"],
                    aneurysm["coordZ"],
                )
                gt_w, gt_h, gt_d = aneurysm["w"], aneurysm["h"], aneurysm["d"]
                center_gt = [gt_x, gt_y, gt_z]
                radius_gt = max(gt_w / 2, gt_h / 2, gt_d / 2)
                matches_aneurysm = []
                for prop in gt_props:
                    min_z, min_y, min_x, max_z, max_y, max_x = prop["bbox"]
                    z_pred, y_pred, x_pred = prop["centroid"]
                    center_pred = [x_pred, y_pred, z_pred]
                    distance_gt = np.linalg.norm(
                        np.array(center_gt) - np.array(center_pred)
                    )
                    radius = max((max_z - min_z, max_y - min_y, max_x - min_x)) / 2
                    # print(distance_gt, radius_gt, radius)
                    if distance_gt < radius_gt + radius:
                        matches_aneurysm.append(True)
                    else:
                        matches_aneurysm.append(False)
                matches_scan.append(matches_aneurysm)
            matches_scan = np.array(matches_scan).astype(int)
            true_positive_count = np.sum(matches_scan, axis=1)
            detected_aneurysms_scan = list(true_positive_count)
            aneurysms_found += detected_aneurysms_scan
            true_positive_count = np.sum(true_positive_count > 0)
            false_positive_count = np.sum(matches_scan, axis=0)
            false_positive_count = np.sum(false_positive_count == 0)
            false_negative_count = len(aneurysms) - true_positive_count
            new_row = [true_positive_count, false_positive_count, false_negative_count]

            rows.append(new_row)
    df_results = pd.DataFrame(rows, columns=cols)
    return df_results, aneurysms_found


def print_metrics(df_results):
    df_results_healthy = df_results[(df_results["tp"] == 0) & (df_results["fn"] == 0)]
    print("Total cases: ", len(df_results))
    print("Healthy cases: ", len(df_results_healthy))
    print("Sick cases: ", len(df_results) - len(df_results_healthy))
    print("FP Rate:", df_results["fp"].sum() / len(df_results))
    print(
        "Recall:",
        df_results["tp"].sum() / (df_results["tp"].sum() + df_results["fn"].sum()),
    )
    print(
        "FP Rate (Healthy patients):",
        df_results_healthy["fp"].sum() / len(df_results_healthy),
    )
    print(
        "Patient-level specificity:",
        len(df_results_healthy[df_results_healthy["fp"] == 0]),
        "/",
        len(df_results_healthy),
    )


def filterfun(text):
    if "Ts" in text:
        return "Int"
    elif "ExtA" in text:
        return "ExtA"
    elif "ExtB" in text:
        return "ExtB"
    else:
        return "Hosp"


def add_size_based_fields(df_labels, spacing=0.4):

    volume_per_voxel = spacing**3
    df_labels["volume_mm"] = df_labels["volume"] * volume_per_voxel
    df_labels["diameter"] = 1.445 * (3 * df_labels["volume_mm"] / (4 * np.pi)) ** (
        1 / 3
    )
    df_labels["size"] = df_labels["diameter"].apply(classify_sizes)
    df_labels["partition"] = df_labels["seriesuid"].apply(filterfun)

    return df_labels


def visualize_iom(preds_fp, mode="arteries"):

    columns = {"arteries": "artery_iom", "veins": "vein_iom"}
    colors = {"arteries": "orange", "veins": "blue"}
    mode_col = columns[mode]
    # plot histogram of intersection for false positives, relative freq
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    ax[0].hist(
        preds_fp[mode_col],
        bins=100,
        range=(0, 1),
        label="False positives",
        density=True,
        color=colors[mode],
    )
    ax[0].set_ylim(len(preds_fp))

    plt.suptitle(f"Percentage of false positive pred. volume intersecting with {mode}")
    # paint the boxplot horizontally
    ax[1].boxplot(
        preds_fp[mode_col],
        vert=False,
    )
    # paint the cumsum of fequencyes
    ax[2].hist(
        preds_fp[mode_col],
        bins=100,
        range=(0, 1),
        cumulative=True,
        density=True,
        label="False positives",
        color=colors[mode],
    )
    ax[0].set_ylim(0, 75)
    ax[0].set_ylabel("Frequency")
    ax[0].set_xlabel("Intersection over Minimum")
    ax[1].set_xlabel("Intersection over Minimum")
    ax[2].set_xlabel("Intersection over Minimum")
    ax[2].set_ylabel("Cumulative sum of relative frequencies")
    return fig
