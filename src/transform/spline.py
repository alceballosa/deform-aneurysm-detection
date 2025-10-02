# -*- coding: utf-8 -*-
from __future__ import division, print_function
import scipy 
from tps import ThinPlateSpline
import random
import torch
import numpy as np
import itk 
from .abstract_transform import AbstractTransform
from skimage import measure, morphology

def get_64_grid_points():
    x = np.linspace(4, 60, 5).astype(int)
    y = np.linspace(4, 60, 5).astype(int)
    z = np.linspace(4, 60, 5).astype(int)

    x, y, z = np.meshgrid(x, y, z)

    # stack as coordinates

    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    return list(grid_points)


def get_transform_points(label, center, amp):
    surface_points = measure.marching_cubes(label, spacing=[1, 1, 1])[0]
    num_points = 200 if len(surface_points) > 200 else len(surface_points)
    indices = np.random.choice(len(surface_points), num_points, replace=False)
    surface_points = surface_points[indices]

    # apply dilation with a ball of radius 3

    radius = 3 * 2 + 1
    selem = np.ones((radius, radius, radius), dtype=bool)

    label_vol_dilated = morphology.binary_dilation(label, footprint=selem)
    surface_points_dilated = measure.marching_cubes(
        label_vol_dilated, spacing=[1, 1, 1]
    )[0].astype(int)
    # randomly and uniformly select 300 points from the dilated surface points
    num_points_dilated = (
        100 if len(surface_points_dilated) > 100 else len(surface_points_dilated)
    )
    indices = np.random.choice(
        len(surface_points_dilated), num_points_dilated, replace=False
    )
    surface_points_dilated = surface_points_dilated[indices]
    
    amp = [amp, amp, amp]

    # dz = amp[0]*2*(np.random.rand(len(surface_points)) - 0.5)
    # dy = amp[1]*2*(np.random.rand(len(surface_points)) - 0.5)
    # dx = amp[2]*2*(np.random.rand(len(surface_points)) - 0.5)

    # random gaussian with mean 0 and std 1
    dz = amp[2] * np.random.normal(0.5, 0.1, len(surface_points))
    dy = amp[1] * np.random.normal(0.5, 0.1, len(surface_points))
    dx = amp[0] * np.random.normal(0.5, 0.1, len(surface_points))



    sign = ((surface_points > center).astype(int)) * 2 - 1
    dz = sign[:, 2] * dz
    dy = sign[:, 1] * dy
    dx = sign[:, 0] * dx
    grid_64 = get_64_grid_points()
    radius = 9 * 2 + 1
    selem = np.ones((radius, radius, radius), dtype=bool)

    label_vol_dilated = morphology.binary_dilation(label, footprint=selem)
    grid_64 = [point for point in grid_64 if label_vol_dilated[tuple(point)] != 1]

    surface_points_change = surface_points + np.stack([dx, dy, dz], axis=1)

    points = np.concatenate(
        [surface_points, surface_points_dilated, center.reshape(-1, 3), grid_64],
        axis=0,
    )
    points_target = np.concatenate(
        [
            surface_points_change,
            surface_points_dilated,
            center.reshape(-1, 3),
            grid_64,
        ],
        axis=0,
    )
    return points, points_target


def transform_with_splines(arrays, source, target, order):
    dim = 3
    spline = itk.ThinPlateSplineKernelTransform[itk.D, dim].New()
    source_landmarks = spline.GetSourceLandmarks()
    source_landmarks.SetPoints(itk.vector_container_from_array(source.flatten()))
    target_landmarks = spline.GetTargetLandmarks()
    target_landmarks.SetPoints(itk.vector_container_from_array(target.flatten()))
    spline.ComputeWMatrix()
    outputs = []
    for array in arrays:
        input_img = itk.image_from_array(array, is_vector=False)
        output_img = itk.resample_image_filter(
            input_img,
            use_reference_image=True,
            reference_image=input_img,
            transform=spline,
        )
        output_array = itk.GetArrayFromImage(output_img)[np.newaxis, ...]

        
        outputs.append(output_array)
    return outputs

def transform_with_splines_2(arrays, source, target, order):
    tps = ThinPlateSpline(0.1, order=order, enforce_tps_kernel=True)
    tps.fit(target, source)
    height, width, depth = arrays[0].shape
    output_indices = np.indices((height, width, depth), dtype=np.float64).transpose(
        1, 2, 3, 0
    )  # Shape: (H, W, 2)
    input_indices = tps.transform(output_indices.reshape(-1, 3)).reshape(
        height, width, depth, 3
    )
    output_arrays = []
    for array in arrays:
        deformed_array = scipy.ndimage.map_coordinates(
            np.array(array), input_indices.transpose(3, 0, 1, 2), order=order, mode="grid-constant"
        )[np.newaxis, ...]
        output_arrays.append(deformed_array)
    return output_arrays

class SplineTransform(AbstractTransform):
    """random flip the image (shape [C, D, H, W] or [C, H, W])"""

    def __init__(self, p=0.5):
        """
        flip_depth (bool) : random flip along depth axis or not, only used for 3D images
        flip_height (bool): random flip along height axis or not
        flip_width (bool) : random flip along width axis or not
        """
        self.p = p

    def __call__(self, sample):
        if len(sample["ctr"]) != 1 or random.random() > self.p:

            return sample
        
        image = sample["image"][0]
        label = sample["label"][0]
        sample["ctr_orig"] = sample["ctr"].copy()
        sample["image_orig"] = sample["image"].copy()
        arrays = [image, label]
        if "mask" in sample:
            mask = sample["mask"][0]
            arrays.append(mask)
        
        if "cvs_mask" in sample:
            cvs_mask = sample["cvs_mask"][0]
            arrays.append(cvs_mask)

        diam =  np.mean(sample["rad"][0])
        amp = ((diam * 2) / 3) / 2 # should be diam * 1
        try:
            points, points_target = get_transform_points(label, sample["ctr"][0], amp)
        except Exception as e:
            print(e)
            print(sample)
            return sample
            
        # rearrange points and points_target as z,x,y
        points = points[:, [2, 1, 0]]
        points_target = points_target[:, [2, 1, 0]]
        arrays_t = transform_with_splines_2(arrays, points, points_target, order=3)
        sample["image"] = arrays_t[0]
        sample["label"] = arrays_t[1]
        if "mask" in sample:
            sample["mask"] = arrays_t[2]
        if "cvs_mask" in sample:
            sample["cvs_mask"] = arrays_t[3]
        
        sample["rad"] = sample["rad"] - amp * 1
        return sample
