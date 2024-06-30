# -*- coding: utf-8 -*-
from __future__ import division, print_function

import random

import numpy as np
import pdb
import SimpleITK as sitk

from .crop import InstanceCrop, reorient
from .vessel_crop import SPInstanceCrop


class InstanceCrop2(InstanceCrop):
    """Randomly crop the input image (shape [C, D, H, W]
    random samples the patches before augmentation to save time
    """

    def __call__(self, sample):
        image = sample["image"].astype("float32")
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
        shape = image.shape

        has_vessel_seg = "mask" in sample.keys()

        if has_vessel_seg:
            vessel = sample["mask"]
            vessel_itk = sitk.GetImageFromArray(vessel)

        

        re_spacing = np.array(self.spacing) / np.array(self.base_spacing)
        crop_size = np.array(self.crop_size) * re_spacing
        overlap = self.overlap * re_spacing

        if self.sample_num > 1:
            if len(instance_loc) > 0:
                num_pos_samples = int(np.ceil(self.sample_num * self.tp_ratio))
            else:
                num_pos_samples = 0  # no positive samples
        else:
            # sample 0 or 1 randomly
            num_pos_samples = np.random.choice([0, 1], p=[1 - self.tp_ratio, self.tp_ratio])
        num_rand_samples = self.sample_num - num_pos_samples

        # print("\nMUESTRAS POSITIVAS", num_pos_samples)
        # print("MUESTRAS ALEATORIASs", num_rand_samples)

        # get center at regular grids
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
        rand_indices = np.random.choice(
            len(crop_centers), size=num_rand_samples, replace=False
        )
        rand_centers = np.array(crop_centers)[rand_indices]

        # get center near objects
        if self.rand_trans is not None:
            instance_crop = instance_loc + np.random.randint(
                low=-self.rand_trans, high=self.rand_trans, size=3
            )
        else:
            instance_crop = instance_loc

        pos_indices = np.random.choice(
            len(instance_crop), size=num_pos_samples, replace=True
        )
        pos_centers = instance_crop[pos_indices]

        all_centers = np.concatenate([pos_centers, rand_centers], axis=0)
        tp_num = []
        all_loc_crops = []
        all_rad_crops = []
        all_cls_crops = []
        matrix_crops = []
        space_crops = []
        for C in all_centers:
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

        CT_crops = []
        if has_vessel_seg:
            vessel_crops = []
        image_spacing_crops = []
        for i in range(len(all_centers)):
            matrix = matrix_crops[i]
            space = space_crops[i]
            image_itk_crop = reorient(
                image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear
            )
            image_crop = sitk.GetArrayFromImage(image_itk_crop)
            CT_crops.append(np.expand_dims(image_crop, axis=0))
            image_spacing_crops.append(space)
            
            if has_vessel_seg:
                vessel_itk_crop = reorient(
                    vessel_itk,
                    matrix,
                    spacing=list(space),
                    interp1=sitk.sitkNearestNeighbor,
                )
                vessel_crop = sitk.GetArrayFromImage(vessel_itk_crop)
                vessel_crops.append(np.expand_dims(vessel_crop, axis=0))


        samples = []
        for i, _ in enumerate(CT_crops):
            ctr = all_loc_crops[i]
            rad = all_rad_crops[i]
            cla = all_cls_crops[i]  # lesion: 0
            shape = np.array(CT_crops[i].shape[1:])

            scale_spacing = image_spacing_crops[i]
            real_space = scale_spacing
            #print(scale_spacing)
            #print(image_spacing)
            #print(rad)
            #print("----")
            # NOTE: aqui se hizo lo de world coordinates
            if len(rad) > 0:
                rad = rad / real_space  # convert pixel coord
            
            sample = {}
            sample["image"] = CT_crops[i]
            sample["ctr"] = ctr
            sample["rad"] = rad
            sample["cls"] = cla
            if has_vessel_seg:
                sample["mask"] = vessel_crops[i]
                sample["volume"] = vessel_crops[i].sum()
            samples.append(sample)

        return samples


class SPInstanceCrop2(SPInstanceCrop):
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

        num_pos_samples = int(np.ceil(self.sample_num * self.tp_ratio))
        num_rand_samples = self.sample_num - num_pos_samples

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
        rand_indices = np.random.choice(
            len(crop_centers), size=num_rand_samples * 10, replace=False
        )
        rand_centers = np.array(crop_centers)[rand_indices]

        if self.rand_trans is not None:
            instance_crop = instance_loc + np.random.randint(
                low=-self.rand_trans, high=self.rand_trans, size=3
            )
        else:
            instance_crop = instance_loc

        pos_indices = np.random.choice(
            len(instance_crop), size=num_pos_samples, replace=True
        )
        pos_centers = instance_crop[pos_indices]

        all_centers = np.concatenate([pos_centers, rand_centers], axis=0)

        tp_num = []
        all_loc_crops = []
        all_rad_crops = []
        all_cls_crops = []
        matrix_crops = []
        space_crops = []
        for C in all_centers:
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

        samples = []
        for i in range(len(all_centers)):
            matrix = matrix_crops[i]
            scale_spacing = space_crops[i]
            image_itk_crop = reorient(
                image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear
            )
            vessel_itk_crop = reorient(
                vessel_itk,
                matrix,
                spacing=list(space),
                interp1=sitk.sitkLinear,
                #interp1=sitk.sitkNearestNeighbor,
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
