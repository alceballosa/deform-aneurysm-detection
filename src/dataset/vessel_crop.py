# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import SimpleITK as sitk

from .crop import InstanceCrop, rand_rot_coord, reorient


class SPInstanceCrop(InstanceCrop):
    """Randomly crop the input image (shape [C, D, H, W]"""

    def __call__(self, sample):
        image = sample["image"].astype("float32")
        vessel = sample["mask"]  ###############
        all_loc = sample["all_loc"]
        all_rad = sample["all_rad"]
        all_cls = sample["all_cls"]
        image_spacing = sample["image_spacing"]
        instance_loc = all_loc[
            np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype="bool")
        ]

        image_itk = sitk.GetImageFromArray(image)
        shadow = np.zeros(image.shape)
        shadow_itk = sitk.GetImageFromArray(shadow)
        vessel_itk = sitk.GetImageFromArray(vessel)
        shape = image.shape

        re_spacing = np.array(self.spacing) / np.array(self.base_spacing)
        crop_size = np.array(self.crop_size) * re_spacing
        overlap = self.overlap * re_spacing

        z_stride = crop_size[0] - overlap[0]
        y_stride = crop_size[1] - overlap[1]
        x_stride = crop_size[2] - overlap[2]

        z_range = np.arange(0, shape[0] - overlap[0], z_stride) + crop_size[0] / 2
        y_range = np.arange(0, shape[1] - overlap[1], y_stride) + crop_size[1] / 2
        x_range = np.arange(0, shape[2] - overlap[2], x_stride) + crop_size[2] / 2

        z_range = np.clip(z_range, a_max=shape[0] - crop_size[0] / 2, a_min=None)
        y_range = np.clip(y_range, a_max=shape[1] - crop_size[1] / 2, a_min=None)
        x_range = np.clip(x_range, a_max=shape[2] - crop_size[2] / 2, a_min=None)

        crop_centers = []
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    crop_centers.append(np.array([z, y, x]))

        if self.instance_crop:
            if self.rand_trans is not None:
                instance_crop = instance_loc + np.random.randint(
                    low=-self.rand_trans, high=self.rand_trans, size=3
                )
            else:
                instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)

        tp_num = []
        all_loc_crops = []
        all_rad_crops = []
        all_cls_crops = []
        matrix_crops = []
        space_crops = []
        for C in crop_centers:
            (
                matrix,
                space,
                all_loc_crop,
                all_rad_crop,
                all_cls_crop,
                n_tp,
            ) = self.make_patch(
                C, re_spacing, crop_size, shadow_itk, all_cls, all_loc, all_rad
            )

            tp_num.append(n_tp)
            all_loc_crops.append(all_loc_crop)
            all_rad_crops.append(all_rad_crop)
            all_cls_crops.append(all_cls_crop)
            matrix_crops.append(matrix)
            space_crops.append(space)

        tp_num = np.array(tp_num)
        tp_idx = tp_num > 0
        neg_idx = tp_num == 0

        if tp_idx.sum() > 0:
            tp_pos = self.tp_ratio / tp_idx.sum()
        else:
            tp_pos = 0

        p = np.zeros(shape=tp_num.shape)
        p[tp_idx] = tp_pos
        p[neg_idx] = (1.0 - p.sum()) / neg_idx.sum() if neg_idx.sum() > 0 else 0
        p = p * 1 / p.sum()

        # sample 10x if needed samples since, some cubes are ignore if it is too sparse
        index = np.random.choice(
            np.arange(len(crop_centers)), size=10 * self.sample_num, p=p
        )

        samples = []
        for i in index:
            matrix = matrix_crops[i]
            scale_spacing = space_crops[i]
            image_itk_crop = reorient(
                image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear
            )
            vessel_itk_crop = reorient(
                vessel_itk,
                matrix,
                spacing=list(space),
                interp1=sitk.sitkNearestNeighbor,
            )  # sitkNearestNeighbor)
            image_crop = sitk.GetArrayFromImage(image_itk_crop)
            vessel_crop = sitk.GetArrayFromImage(vessel_itk_crop)
            if vessel_crop[10:-10, 10:-10, 10:-10].sum() < 2000:
                continue

            ct_crop = np.expand_dims(image_crop, axis=0)
            vessel_crop = np.expand_dims(vessel_crop, axis=0)

            ctr = all_loc_crops[i]
            rad = all_rad_crops[i]
            cls = all_cls_crops[i]  # lesion: 0
            shape = np.array(ct_crop.shape[1:])

            real_space = image_spacing * scale_spacing
            if len(rad) > 0:
                rad = rad / real_space  # convert pixel coord
            sample = {}
            sample["image"] = ct_crop
            sample["mask"] = vessel_crop
            sample["volume"] = vessel_crop.sum()
            sample["ctr"] = ctr
            sample["rad"] = rad
            sample["cls"] = cls
            samples.append(sample)
            if len(samples) == self.sample_num:
                break
        return samples
