import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from skimage import morphology


def remove_by_roi(preds, dataset_name):
    if dataset_name == "cmha":
        folder_brain = Path(
            "/projects/vig/Datasets/aneurysm/cta_datasets/cmha/crop_0.4_totalseg/"
        )
        folder_cvs = Path("/projects/vig/Datasets/aneurysm/cta_datasets/cmha/cvs_bbox/")
        all_seriesuid = preds["seriesuid"].unique()
        selected_preds = []
        for seriesuid in all_seriesuid:
            print(f"Processing {seriesuid}")
            preds_seriesuid = preds[preds["seriesuid"] == seriesuid]
            if len(preds_seriesuid) == 0:
                continue
            # load the bbox with simpleitk
            path_bbox = folder_cvs / f"{seriesuid}"
            bbox = sitk.ReadImage(str(path_bbox))
            bbox = sitk.GetArrayFromImage(bbox)
            path_brain = folder_brain / f"{seriesuid.split('.')[0]}/brain.nii.gz"
            brain = sitk.ReadImage(str(path_brain))
            brain = sitk.GetArrayFromImage(brain)
            assert bbox.shape == brain.shape
            brain_plus_bbox = np.logical_or(bbox, brain)

            # iterate over each bbox
            for i, row in preds_seriesuid.iterrows():
                box = np.array(
                    row[["coordX", "coordY", "coordZ", "w", "h", "d"]]
                ).astype(int)
                # add one axis
                box = np.expand_dims(box, axis=0)

                box = xyzwhd2xyzxyz(torch.tensor(box))
                box = box.numpy()
                # get the box in the bbox
                box = box.astype(int)[0]
                x = max(0, box[0])
                y = max(0, box[1])
                z = max(0, box[2])
                x_2 = min(bbox.shape[2], box[3])
                y_2 = min(bbox.shape[1], box[4])
                z_2 = min(bbox.shape[0], box[5])
                # check if the box is in the bbox
                # if np.sum(brain_plus_bbox[z:z_2, y:y_2, x:x_2]) > 0:
                row["overlap"] = np.sum(brain_plus_bbox[z:z_2, y:y_2, x:x_2]) / (
                    (x_2 - x) * (y_2 - y) * (z_2 - z)
                )
                selected_preds.append(row)
        len_before = len(preds)
        preds = pd.DataFrame(selected_preds)
        len_after = len(preds)
        print(f"Before: {len_before}, After: {len_after}")
    return preds


def xyzwhd2xyzxyz(boxes):
    res = torch.zeros_like(boxes)
    res[:, :3] = boxes[:, :3] - boxes[:, 3:] / 2
    res[:, 3:] = boxes[:, :3] + boxes[:, 3:] / 2
    return res


if __name__ == "__main__":
    # Define the input and output directories
    iou_thrs = [0.2, 0.3]

    # get exp from command line arg
    exp_base = Path(sys.argv[1])
    print(exp_base)
    dataset_name = exp_base.name
    exps = [x for x in exp_base.glob("*")]
    for exp_dir in exps:

        # get all dirs starting with "inference_"
        inf_appends = sorted(
            [x.name.replace("inference_", "") for x in exp_dir.glob("inference_*")]
        )

        for inf_append in inf_appends:
            path_inf = "inference_" + inf_append

            for iou_thr in iou_thrs:

                print(f"Filtering iou_thr: {iou_thr} at {inf_append}")
                path_roi = path_preds = (
                    exp_dir / f"inference_{inf_append}" / "predict_roi.csv"
                )
                # check if exists
                # if path_roi.exists():
                #    print(f"Already exists: {path_roi}")
                #    continue
                n_workers = 8
                out_dir = exp_dir / f"iou{iou_thr:.1f}_froc_{inf_append}"
                path_preds = exp_dir / f"inference_{inf_append}" / "predict.csv"
                try:
                    preds = pd.read_csv(path_preds)
                except:
                    print(f"File not found: {path_preds}")
                    continue
                preds = remove_by_roi(preds, dataset_name)
                # save under same location with name "predict_roi.csv"

                preds.to_csv(path_roi, index=False)
                print(f"Saved to {path_roi}")
