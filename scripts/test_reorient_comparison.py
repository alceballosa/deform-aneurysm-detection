"""
Visual test: compare annotation center from DetectionCropper vs actual
center of the lesion object after the same spatial transform.

Approach:
  1. Draw a solid bounding box in the full-size volume at the annotation location.
  2. Pass it as "vessel_edt" alongside the image so DetectionCropper applies
     the exact same transform (matrix, spacing) to both.
  3. From the cropped label, extract the object center using scipy.ndimage.
  4. Compare that against the cropper's reported annotation center (ctr).
"""

import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset.crop2 import DetectionCropper


def normalize(data, window=(0.0, 800.0)):
    min_v, max_v = window
    data = np.clip(data, min_v, max_v)
    return (data - (min_v + max_v) / 2) / ((max_v - min_v) / 2)


def draw_solid_box(shape, center_zyx, rad_zyx):
    """Create a binary volume with a solid box at the given center/radii."""
    label = np.zeros(shape, dtype=np.float32)
    ctr = np.round(center_zyx).astype(int)
    half = np.round(rad_zyx / 2).astype(int)

    z0 = max(0, ctr[0] - half[0])
    z1 = min(shape[0] - 1, ctr[0] + half[0])
    y0 = max(0, ctr[1] - half[1])
    y1 = min(shape[1] - 1, ctr[1] + half[1])
    x0 = max(0, ctr[2] - half[2])
    x1 = min(shape[2] - 1, ctr[2] + half[2])

    label[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = 1.0
    return label


def main():
    scan_path = (
        "/projects/vig/Datasets/aneurysm/cta_datasets/"
        "internal_train/crop_0.4/Tr0001.nii.gz"
    )
    output_dir = (
        "/projects/vig/alberto/medical/exploration/deform/"
        "scripts/reorient_comparison"
    )
    os.makedirs(output_dir, exist_ok=True)

    crop_size = [128, 128, 128]
    spacing_range = [0.9, 1.8]

    # Real annotation for Tr0001
    # coordX=253, coordY=231, coordZ=141, w=19, h=21, d=13
    # In (z, y, x) order:
    annot_loc = np.array([[141.0, 231.0, 253.0]])
    annot_rad = np.array([[13.0, 21.0, 19.0]])  # d, h, w
    annot_cls = np.array([0], dtype="int8")  # aneurysm

    print(f"Loading {scan_path}")
    image_header = sitk.ReadImage(scan_path)
    image_arr = sitk.GetArrayFromImage(image_header).astype("float32")
    image_arr = normalize(image_arr)
    print(f"  Volume shape: {image_arr.shape}")

    # Draw solid box in full-size volume at the annotation location
    label_full = draw_solid_box(image_arr.shape, annot_loc[0], annot_rad[0])
    print(f"  Label box voxel count: {int(label_full.sum())}")

    # Build DetectionCropper with padded_reorient=False, no rotation/translation
    cropper = DetectionCropper(
        crop_size=crop_size,
        rand_trans=None,
        rand_rot=None,
        rand_space=spacing_range,
        spacing=[1.0, 1.0, 1.0],
        overlap=[16, 32, 32],
        tp_ratio=1.0,
        sample_num=3,
        blank_side=0,
        padded_reorient=False,
        sample_cls=[0],
    )

    # Pass label as "vessel_edt" so it gets the same transform as the image
    sample = {
        "scan_id": "Tr0001",
        "image": image_arr,
        "image_spacing": (1.0, 1.0, 1.0),
        "all_loc": annot_loc,
        "all_rad": annot_rad,
        "all_cls": annot_cls,
        "vessel_edt": label_full,
    }

    np.random.seed(42)
    patches = cropper(sample)

    for i, patch in enumerate(patches):
        crop_arr = patch["image"][0]  # (1, D, H, W) -> (D, H, W)
        label_crop = patch["vessel_edt"][0]  # same transform applied
        crop_shape = crop_arr.shape

        # 1. Center from DetectionCropper annotations
        ctr = np.array(patch["ctr"])
        rad = np.array(patch["rad"])
        if len(ctr) > 0:
            cropper_ctr = ctr[0]
            cropper_rad = rad[0]
        else:
            cropper_ctr = np.array([np.nan, np.nan, np.nan])
            cropper_rad = np.array([np.nan, np.nan, np.nan])

        # 2. Center from the actual transformed label volume
        if label_crop.sum() > 0:
            label_com = np.array(ndimage.center_of_mass(label_crop))
            # Also get bounding box to extract center from bbox
            labeled, _ = ndimage.label(label_crop > 0.5)
            slices = ndimage.find_objects(labeled)[0]
            bbox_ctr = np.array([
                (slices[0].start + slices[0].stop) / 2.0,
                (slices[1].start + slices[1].stop) / 2.0,
                (slices[2].start + slices[2].stop) / 2.0,
            ])
        else:
            label_com = np.array([np.nan, np.nan, np.nan])
            bbox_ctr = np.array([np.nan, np.nan, np.nan])

        print(f"\nTrial {i}: crop shape={crop_shape}")
        print(f"  Cropper annot ctr  = {np.round(cropper_ctr, 1)}")
        print(f"  Label center-of-mass = {np.round(label_com, 1)}")
        print(f"  Label bbox center  = {np.round(bbox_ctr, 1)}")
        print(f"  Diff (cropper - COM) = {np.round(cropper_ctr - label_com, 1)}")

        # Save outputs for visual inspection
        crop_itk = sitk.GetImageFromArray(crop_arr)
        img_path = os.path.join(output_dir, f"trial{i}_image.nii.gz")
        sitk.WriteImage(crop_itk, img_path)

        label_itk = sitk.GetImageFromArray(label_crop)
        label_itk.CopyInformation(crop_itk)
        label_path = os.path.join(output_dir, f"trial{i}_label.nii.gz")
        sitk.WriteImage(label_itk, label_path)

        print(f"  Saved: {os.path.basename(img_path)}, "
              f"{os.path.basename(label_path)}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
