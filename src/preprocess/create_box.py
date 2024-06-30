import os
import pdb
import random

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import tqdm
from scipy.ndimage import label


def GetBoundingBox_From_Coords(coordinates, img_header) -> tuple:
    """
    coords is a list in [[z, y, x], [z, y, x], ....] type
    spacing is a list for x, y, z order
    """
    coordinates = np.array(coordinates)
    z, y, x = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    z_min, z_max = min(z), max(z)
    y_min, y_max = min(y), max(y)
    x_min, x_max = min(x), max(x)
    center_z, center_y, center_x = (
        (z_min + z_max) / 2,
        (y_min + y_max) / 2,
        (x_min + x_max) / 2,
    )
    d, h, w = (
        (z_max - z_min + 1),  # * z_spacing,
        (y_max - y_min + 1),  # * y_spacing,
        (x_max - x_min + 1),  # * x_spacing,
    )
    return center_z, center_y, center_x, d, h, w


def GetBoundingBox_From_Coords_World(coordinates, img_header) -> tuple:
    """
    coords is a list in [[z, y, x], [z, y, x], ....] type
    spacing is a list for x, y, z order
    """
    coordinates = np.array(coordinates)
    z, y, x = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    x_spacing, y_spacing, z_spacing = img_header.GetSpacing()
    z_min, z_max = min(z), max(z)
    y_min, y_max = min(y), max(y)
    x_min, x_max = min(x), max(x)
    center_z, center_y, center_x = (
        (z_min + z_max) / 2,
        (y_min + y_max) / 2,
        (x_min + x_max) / 2,
    )
    center_x, center_y, center_z = img_header.TransformContinuousIndexToPhysicalPoint(
        [center_x, center_y, center_z]
    )
    d, h, w = (
        (z_max - z_min + 1) * z_spacing,
        (y_max - y_min + 1) * y_spacing,
        (x_max - x_min + 1) * x_spacing,
    )
    return center_z, center_y, center_x, d, h, w


if __name__ == "__main__":
    mode = "internal_train"
    # train_image_path = 'imagesTs'
    # train_label_path = "/work/vig/Datasets/aneurysm/external_label"
    # train_label_path = "/work/vig/Datasets/aneurysm/hospital/crop_0.4_label"
    # train_label_path = "/work/vig/Datasets/aneurysm/hospital/crop_0.4_label"
    # train_label_path = "/work/vig/Datasets/aneurysm/internal_label_test"
    # train_label_path = "/work/vig/Datasets/aneurysm/internal_label_train"
    # train_label_path = "/work/vig/Datasets/aneurysm/internal_train/crop_0.4_label"
    label_path = f"/work/vig/Datasets/aneurysm/{mode}/crop_0.4_label"
    vessel_path = f"/work/vig/Datasets/aneurysm/{mode}/crop_0.4_vessel_v2"
    image_list = os.listdir(label_path)
    data_information = []
    for i in tqdm.tqdm(image_list):
        # image = sitk.ReadImage(os.path.join(train_image_path, i)) # x, y, z
        # spacing = image.GetSpacing()
        mask = sitk.ReadImage(os.path.join(label_path, i))

        mask_arr = sitk.GetArrayFromImage(mask)

        labeled_array, num_features = label(
            mask_arr, structure=ndimage.generate_binary_structure(3, 3)
        )

        if num_features == 0:
            continue
        vessel_header = sitk.ReadImage(os.path.join(vessel_path, i))
        vessel_arr = sitk.GetArrayFromImage(vessel_header)

        artery_arr = (vessel_arr == 1).astype(np.uint8)
        vein_arr = (vessel_arr == 2).astype(np.uint8)
        region = measure.regionprops(labeled_array)

        for j in range(num_features):
            coords = region[j].coords
            z, y, x, d, h, w = GetBoundingBox_From_Coords(coords, mask)
            volume = region[j].area
            try:
                min_axis = region[j].axis_minor_length
            except Exception:
                min_axis = 1
            maj_axis = region[j].axis_major_length
            assert int(volume) == (labeled_array == (j + 1)).sum()
            volume_array = (labeled_array == (j + 1)).astype(np.uint8)
            area_aneurysm = np.sum(volume_array)
            try:
                area_intersection_artery = np.sum(
                    np.logical_and(volume_array, artery_arr)
                )
                area_intersection_vein = np.sum(np.logical_and(volume_array, vein_arr))
            except:
                print(i)
                if i == "ExtA0032.nii.gz":
                    area_intersection_artery = np.sum(
                        np.logical_and(volume_array[:, 119:, 119:], artery_arr)
                    )
                    area_intersection_vein = np.sum(
                        np.logical_and(volume_array[:, 119:, 119:], vein_arr)
                    )
                elif i == "ExtB0042.nii.gz":
                    area_intersection_artery = np.sum(
                        np.logical_and(volume_array, artery_arr[:-1, :-1, :-1])
                    )
                    area_intersection_vein = np.sum(
                        np.logical_and(volume_array, vein_arr[:-1, :-1, :-1])
                    )
                elif i == "ExtB0061.nii.gz":
                    area_intersection_artery = np.sum(
                        np.logical_and(volume_array, artery_arr[:-1, :-1, :-1])
                    )
                    area_intersection_vein = np.sum(
                        np.logical_and(volume_array, vein_arr[:-1, :-1, :-1])
                    )
                else:
                    area_intersection_artery = np.sum(
                        np.logical_and(volume_array, artery_arr[:, :-1, :-1])
                    )
                    area_intersection_vein = np.sum(
                        np.logical_and(volume_array, vein_arr[:, :-1, :-1])
                    )
            iom_artery = area_intersection_artery / area_aneurysm
            iom_vein = area_intersection_vein / area_aneurysm
            data_information.append(
                [
                    i,
                    x,
                    y,
                    z,
                    w,
                    h,
                    d,
                    "aneurysm",
                    volume,
                    min_axis,
                    maj_axis,
                    iom_artery,
                    iom_vein,
                ]
            )
            print(data_information[-1])

    df = pd.DataFrame(
        data=data_information,
        columns=[
            "seriesuid",
            "coordX",
            "coordY",
            "coordZ",
            "w",
            "h",
            "d",
            "lesion",
            "volume",
            "min_axis",
            "maj_axis",
            "iom_artery",
            "iom_vein",
        ],
    )
    df.to_csv(f"./labels/gt/{mode}_crop_0.4.csv", index=False)
    # df.to_csv("/home/ceballosarroyo.a/workspace/medical/cta-det2/labels/hospital_0.4.csv", index=False)
    # df.to_csv('test.csv', index=False)
    # df.to_csv('train.csv', index=False)
    # df.to_csv('train0.4.csv', index=False)
    # df.to_csv('test0.4.csv', index=False)
