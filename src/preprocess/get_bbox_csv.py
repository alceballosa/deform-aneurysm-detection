import os
import sys

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
    print(sys.argv)
    path_label = sys.argv[1]
    path_save = sys.argv[2]

    image_list = os.listdir(path_label)
    list_aneurysms = []
    for i in tqdm.tqdm(image_list):
        mask = sitk.ReadImage(os.path.join(path_label, i))
        mask_arr = sitk.GetArrayFromImage(mask)
        mask_arr = mask_arr + np.abs(np.min(mask_arr))
        labeled_array, num_features = label(
            mask_arr, structure=ndimage.generate_binary_structure(3, 3)
        )
        if num_features == 0:
            continue

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

            list_aneurysms.append(
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
                ]
            )
            print(list_aneurysms[-1])

    df = pd.DataFrame(
        data=list_aneurysms,
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
        ],
    )
    df.to_csv(path_save, index=False)
