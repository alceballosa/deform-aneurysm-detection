import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi
from skimage import morphology

mask_cache = {
    "artery": {},
    "vein": {},
    "cvs_bbox": {},
    "brain": {},
    "cvs_mask": {},
}


# steps:

# 1. Load ARTERY MASK
# 2. Load VEIN MASK
# 3. Load CVS MASK
# 4. Load BRAIN MASK
# 5. Dilate BRAIN MASK by 9 voxels
# 6. Add CVS BBOX to BRAIN MASK to get enhanced BRAIN MASK
# 7. VEIN MASK = VEIN MASK - CVS MASK


def apply_fp_removal(preds, dataset_name):
    path_root = Path("/projects/vig/Datasets/aneurysm/cta_datasets")

    if dataset_name == "cta_rsna_ane":
        path_root = Path("/scratch/ceballosarroyo.a/aneurysm/mm_datasets/")

    path_vessels = path_root / f"{dataset_name}/crop_0.4_vessel/"
    path_cvs_bbox = path_root / f"{dataset_name}/cvs_bbox/"
    path_cvs_mask = path_root / f"{dataset_name}/cvs_mask/"
    path_brain = path_root / f"{dataset_name}/crop_0.4_totalseg/"

    selected_preds = []
    all_seriesuid = preds["seriesuid"].unique()

    for i, seriesuid in enumerate(all_seriesuid):
        print(f"Processing series {i}: {seriesuid}")
        preds_seriesuid = preds[preds["seriesuid"] == seriesuid]
        if len(preds_seriesuid) == 0:
            continue

        if seriesuid in mask_cache["artery"]:
            artery_mask = mask_cache["artery"][seriesuid]
            # cvs_bbox = mask_cache["cvs_bbox"][seriesuid]
            # cvs_mask = mask_cache["cvs_mask"][seriesuid]
            vein_mask = mask_cache["vein"][seriesuid]
            enhanced_brain_mask = mask_cache["brain"][seriesuid]
        else:
            # Load masks
            try:
                #print(str(path_vessels / f"{seriesuid}"))
                vessel_mask = sitk.ReadImage(str(path_vessels / f"{seriesuid}"))
            except:
                print(f"Vessel mask not found for {seriesuid}, skipping whole thing")
                return
            vessel_mask = sitk.GetArrayFromImage(vessel_mask)

            cvs_bbox = sitk.ReadImage(str(path_cvs_bbox / f"{seriesuid}"))
            cvs_bbox = sitk.GetArrayFromImage(cvs_bbox).astype(bool)

            cvs_mask = sitk.ReadImage(str(path_cvs_mask / f"{seriesuid}"))
            cvs_mask = sitk.GetArrayFromImage(cvs_mask).astype(bool)

            brain_mask = sitk.ReadImage(
                str(path_brain / f"{seriesuid.replace('.nii.gz', '')}" / "brain.nii.gz")
            )
            brain_mask = sitk.GetArrayFromImage(brain_mask)

            # Dilate brain mask
            diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
            brain_mask = ndi.binary_dilation(brain_mask, diamond, iterations=9)

            # Enhance brain mask
            enhanced_brain_mask = np.logical_or(brain_mask, cvs_bbox).astype(bool)

            artery_mask = (vessel_mask == 1).astype(bool)
            vein_mask = (vessel_mask == 2).astype(bool)

            # remove CVS from vein mask with logical op

            vein_mask = np.logical_and(vein_mask, np.logical_not(cvs_mask)).astype(bool)

            # Cache masks

            #mask_cache["artery"][seriesuid] = artery_mask
            #mask_cache["vein"][seriesuid] = vein_mask
            # mask_cache["cvs_bbox"][seriesuid] = cvs_bbox
            # mask_cache["cvs_mask"][seriesuid] = cvs_mask
            #mask_cache["brain"][seriesuid] = enhanced_brain_mask

        # check if exists

        # Iterate over predictions
        for i, row in preds_seriesuid.iterrows():
            box = np.array(row[["coordX", "coordY", "coordZ", "w", "h", "d"]]).astype(
                int
            )
            # add one axis
            box = np.expand_dims(box, axis=0)

            box = xyzwhd2xyzxyz(torch.tensor(box))
            box = box.numpy()
            # get the box in the bbox
            box = box.astype(int)[0]
            x = max(0, box[0])
            y = max(0, box[1])
            z = max(0, box[2])
            x_2 = min(enhanced_brain_mask.shape[2], box[3])
            y_2 = min(enhanced_brain_mask.shape[1], box[4])
            z_2 = min(enhanced_brain_mask.shape[0], box[5])

            # Calculate overlap with enhanced brain mask
            row["overlap_enhanced_brain"] = np.sum(
                enhanced_brain_mask[z:z_2, y:y_2, x:x_2]
            ) / ((x_2 - x) * (y_2 - y) * (z_2 - z))

            # Calculate overlap with artery and vein masks
            row["overlap_artery"] = np.sum(artery_mask[z:z_2, y:y_2, x:x_2]) / (
                (x_2 - x) * (y_2 - y) * (z_2 - z)
            )
            row["overlap_vein"] = np.sum(vein_mask[z:z_2, y:y_2, x:x_2]) / (
                (x_2 - x) * (y_2 - y) * (z_2 - z)
            )

            selected_preds.append(row)
            assert vein_mask.shape == enhanced_brain_mask.shape
            # assert cvs_mask.shape == enhanced_brain_mask.shape
            # assert cvs_bbox.shape == enhanced_brain_mask.shape

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
            path_roi = path_preds = (
                exp_dir / f"inference_{inf_append}" / "predict_roi_jisoo.csv"
            )
            # check if exists
            if path_roi.exists():
                print(f"Already exists: {path_roi}")
                continue
            n_workers = 8
            path_preds = exp_dir / f"inference_{inf_append}" / "predict.csv"
            try:
                print(f"Doing {path_preds} ")
                preds = pd.read_csv(path_preds)
            except:
                print(f"File not found: {path_preds}")
                continue
            preds = apply_fp_removal(preds, dataset_name)
            if not (preds is None):
                # save under same location with name "predict_roi.csv"

                preds.to_csv(path_roi, index=False)
                print(f"Saved to {path_roi}")
            else:
                print(f"No predictions to save for {path_roi}, review previous steps.")
