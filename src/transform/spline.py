# -*- coding: utf-8 -*-
from __future__ import division, print_function

import random

import itk
import numpy as np
import scipy
import torch
from skimage import measure, morphology
# from torch_tps import ThinPlateSpline as ThinPlateSplineGPU
# from tps import ThinPlateSpline

from .abstract_transform import AbstractTransform

# def transform_with_splines(arrays, source, target, order):
#     dim = 3
#     spline = itk.ThinPlateSplineKernelTransform[itk.D, dim].New()
#     source_landmarks = spline.GetSourceLandmarks()
#     source_landmarks.SetPoints(itk.vector_container_from_array(source.flatten()))
#     target_landmarks = spline.GetTargetLandmarks()
#     target_landmarks.SetPoints(itk.vector_container_from_array(target.flatten()))
#     spline.ComputeWMatrix()
#     outputs = []
#     for array in arrays:
#         input_img = itk.image_from_array(array, is_vector=False)
#         output_img = itk.resample_image_filter(
#             input_img,
#             use_reference_image=True,
#             reference_image=input_img,
#             transform=spline,
#         )
#         output_array = itk.GetArrayFromImage(output_img)[np.newaxis, ...]


#         outputs.append(output_array)
#     return outputs

# def transform_with_splines_2(arrays, source, target, order):
#     tps = ThinPlateSpline(0.1, order=order, enforce_tps_kernel=True)
#     tps.fit(target, source)
#     height, width, depth = arrays[0].shape
#     output_indices = np.indices((height, width, depth), dtype=np.float64).transpose(
#         1, 2, 3, 0
#     )  # Shape: (H, W, 2)
#     input_indices = tps.transform(output_indices.reshape(-1, 3)).reshape(
#         height, width, depth, 3
#     )
#     output_arrays = []
#     for array in arrays:
#         deformed_array = scipy.ndimage.map_coordinates(
#             np.array(array), input_indices.transpose(3, 0, 1, 2), order=order, mode="grid-constant"
#         )[np.newaxis, ...]
#         output_arrays.append(deformed_array)
#     return output_arrays


# def transform_with_splines_gpu(arrays, source, target, order):
#     # convert arrays to tensors
#     print(source, target)
#     arrays = [torch.from_numpy(array).float().to("cuda:0") for array in arrays]
#     source = torch.from_numpy(source).float().to("cuda:0")
#     target = torch.from_numpy(target).float().to("cuda:0")

#     tps = ThinPlateSplineGPU(0.1, order=order, enforce_tps_kernel=True, device="cuda:0")
#     tps.fit(target, source)
#     height, width, depth = arrays[0].size()

#     i = torch.arange(height, dtype=torch.float32)
#     j = torch.arange(width, dtype=torch.float32)
#     k = torch.arange(depth, dtype=torch.float32)
#     ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')
#     output_indices = torch.cat((ii[..., None], jj[..., None], kk[..., None]), dim=-1).to("cuda:0")  # Shape: (H, W, D, 3)
#     input_indices = tps.transform(output_indices.reshape(-1, 3)).reshape(
#         height, width, depth, 3
#     )
#     grid = input_indices.clone()
#     grid[..., 0] = 2.0 * grid[..., 0] / (
#         height - 1
#     ) - 1.0
#     grid[..., 1] = 2.0 * grid[..., 1] / (
#         width - 1
#     ) - 1.0
#     grid[..., 2] = 2.0 * grid[..., 2] / (
#         depth - 1
#     ) - 1.0


#     newgrid = grid.clone()
#     newgrid[..., 0] = grid[..., 2]
#     newgrid[..., 1] = grid[..., 1]
#     newgrid[..., 2] = grid[..., 0]

#     output_arrays = []
#     for array in arrays:
#         array = array.unsqueeze(0).unsqueeze(0)
#         deformed_array = torch.nn.functional.grid_sample(array, newgrid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=False)
#         deformed_array = deformed_array.squeeze(0).cpu().numpy()
#         output_arrays.append(deformed_array)
#     return output_arrays


def ensure_bounds(points, shape):
    points[:, 0] = np.clip(points[:, 0], 0, shape[0] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, shape[1] - 1)
    points[:, 2] = np.clip(points[:, 2], 0, shape[2] - 1)
    return points


def sample_along_line(p, center, dist):
    # Given a point p, a center point, and a distance dist, return a new point that is dist away from p along the line connecting p and center
    direction = center - p
    direction_norm = np.linalg.norm(direction, axis=1)
    direction_unit = direction / direction_norm.reshape(-1, 1)
    new_point = p + direction_unit * dist
    return new_point


def get_grid_points():
    x = np.linspace(4, 60, 5).astype(int)
    y = np.linspace(4, 60, 5).astype(int)
    z = np.linspace(4, 60, 5).astype(int)
    x, y, z = np.meshgrid(x, y, z)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    return list(grid_points)


def filter_border_points(points, shape, margin=1):
    mask = (
        (points[:, 0] > margin)
        & (points[:, 0] < shape[0] - margin)
        & (points[:, 1] > margin)
        & (points[:, 1] < shape[1] - margin)
        & (points[:, 2] > margin)
        & (points[:, 2] < shape[2] - margin)
    )
    return mask


def touches_border(label):
    # check if a binary label touches the border of the volume
    if (
        np.any(label[0, :, :])
        or np.any(label[-1, :, :])
        or np.any(label[:, 0, :])
        or np.any(label[:, -1, :])
        or np.any(label[:, :, 0])
        or np.any(label[:, :, -1])
    ):
        return True
    else:
        return False


class SplineTransform(AbstractTransform):
    """Use a spline to shrink the aneurysm surface towards the center, while keeping the rest of the volume mostly unchanged"""

    def __init__(
        self,
        p=0.5,
        order=2,
        shrink_factor=8,
        max_points_surf=200,
        max_points_dilated=100,
        use_grid_controls=True,
        device="cuda",
    ):
        """
        Args:
            p (float): probability of applying the transform.
            order (int): The order of the spline interpolation, default is 3.
            shrink_factor (float): Factor to determine how much to shrink surface points towards the center, as a divisor of minor axis length
            max_points_surf (int): Maximum number of surface points to sample.
            max_points_dilated (int): Maximum number of dilated surface points to sample.
            use_grid_controls (bool): Whether to use grid points outside the dilated label as control points.
            device (str): "cpu" or "cuda"
        """
        self.p = p
        self.order = order
        self.shrink_factor = shrink_factor
        self.max_points_surf = max_points_surf
        self.max_points_dilated = max_points_dilated
        self.use_grid_controls = use_grid_controls
        self.device = device
        if device == "cpu":
            self.fun_spline = self.transform_with_splines_cpu
        elif device == "cuda":

            self.fun_spline = self.transform_with_splines_gpu

    def __call__(self, sample):
        if (
            len(sample["ctr"]) != 1
            or random.random() > self.p
            or touches_border(sample["label"][0])
            or np.min(sample["rad"]) < 4.5
            or np.max(sample["rad"]) > 20
        ):
            return sample
        image = sample["image"][0]
        label = sample["label"][0]
        arrays = [image, label]
        if "mask" in sample:
            mask = sample["mask"][0]
            arrays.append(mask)

        if "cvs_mask" in sample:
            cvs_mask = sample["cvs_mask"][0]
            arrays.append(cvs_mask)

        try:
            points, points_target, amp = self.get_transform_points(label)
        except Exception as e:
            print(e)
            print(sample)
            return sample

        arrays_t = self.fun_spline(arrays, points, points_target)
        sample["image"] = arrays_t[0]
        sample["label"] = arrays_t[1]
        if self.device == "cuda":
            sample["label"] = (sample["label"] > 0.66).astype(int)

        if "mask" in sample:
            sample["mask"] = arrays_t[2]
        if "cvs_mask" in sample:
            sample["cvs_mask"] = arrays_t[3]

        sample["rad"] = sample["rad"] - np.mean(amp) * 2

        return sample

    def transform_with_splines_cpu(self, arrays, source, target):
        tps = ThinPlateSpline(0.5, order=self.order, enforce_tps_kernel=True)
        tps.fit(target, source)
        height, width, depth = arrays[0].shape
        output_indices = np.indices((height, width, depth), dtype=np.float64).transpose(
            1, 2, 3, 0
        )
        input_indices = tps.transform(output_indices.reshape(-1, 3)).reshape(
            height, width, depth, 3
        )
        output_arrays = []
        for array in arrays:
            deformed_array = scipy.ndimage.map_coordinates(
                np.array(array),
                input_indices.transpose(3, 0, 1, 2),
                order=self.order,
                mode="grid-constant",
            )[np.newaxis, ...]
            output_arrays.append(deformed_array)
        return output_arrays

    def transform_with_splines_gpu(self, arrays, source, target):
        # convert arrays to tensors

        id_gpu = int(np.random.randint(0, 2))

        arrays = [
            torch.from_numpy(array).float().to(f"cuda:{id_gpu}") for array in arrays
        ]
        source = torch.from_numpy(source).float().to(f"cuda:{id_gpu}")
        target = torch.from_numpy(target).float().to(f"cuda:{id_gpu}")

        tps = ThinPlateSplineGPU(
            0.1, order=self.order, enforce_tps_kernel=True, device=f"cuda:{id_gpu}"
        )
        tps.fit(target, source)
        height, width, depth = arrays[0].size()

        i = torch.arange(height, dtype=torch.float32)
        j = torch.arange(width, dtype=torch.float32)
        k = torch.arange(depth, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(i, j, k, indexing="ij")
        output_indices = torch.cat(
            (ii[..., None], jj[..., None], kk[..., None]), dim=-1
        ).to(
            f"cuda:{id_gpu}"
        )  # Shape: (H, W, D, 3)
        input_indices = tps.transform(output_indices.reshape(-1, 3)).reshape(
            height, width, depth, 3
        )
        grid = input_indices.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / (height - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (width - 1) - 1.0
        grid[..., 2] = 2.0 * grid[..., 2] / (depth - 1) - 1.0

        newgrid = grid.clone()
        newgrid[..., 0] = grid[..., 2]
        newgrid[..., 1] = grid[..., 1]
        newgrid[..., 2] = grid[..., 0]

        output_arrays = []
        for array in arrays:
            array = array.unsqueeze(0).unsqueeze(0)
            deformed_array = torch.nn.functional.grid_sample(
                array,
                newgrid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            deformed_array = deformed_array.squeeze(0).cpu().numpy()
            output_arrays.append(deformed_array)
        return output_arrays

    def get_transform_points(self, label):
        surface_points = measure.marching_cubes(label, spacing=[1, 1, 1])[0]
        num_points = (
            self.max_points_surf
            if len(surface_points) > self.max_points_surf
            else len(surface_points)
        )
        indices = np.random.choice(len(surface_points), num_points, replace=False)
        surface_points = surface_points[indices]
        feats = measure.regionprops(label.astype(int))
        center = np.array(feats[0].centroid)
        # shrink by a random fraction of minor axis length, centered around a shrink factor
        shrink_factor = np.random.randint(
            self.shrink_factor - 1, self.shrink_factor + 1
        )
        #minor_axis_factor = np.random.normal(
        #    shrink_factor, 0.25, len(surface_points)
        #).reshape(-1, 1)
        minor_axis_factor = shrink_factor
        amp = feats[0].axis_minor_length / minor_axis_factor
        surface_points_change = sample_along_line(surface_points, center, amp)
        # apply mask to keep border points unchanged
        mask_surface_points = filter_border_points(
            surface_points, label.shape, margin=3
        )
        surface_points_change[~mask_surface_points] = surface_points[
            ~mask_surface_points
        ]

        # sample control points from dilated surface
        selem = np.ones((7, 7, 7), dtype=bool)
        label_dilated = morphology.binary_dilation(label, footprint=selem)
        surface_points_dilated = measure.marching_cubes(
            label_dilated, spacing=[1, 1, 1]
        )[0].astype(int)
        num_points_dilated = (
            self.max_points_dilated
            if len(surface_points_dilated) > self.max_points_dilated
            else len(surface_points_dilated)
        )
        indices = np.random.choice(
            len(surface_points_dilated), num_points_dilated, replace=False
        )
        surface_points_dilated = ensure_bounds(
            surface_points_dilated[indices], label.shape
        )

        # sample a grid outside the dilated label as additional control points
        # for the rest of the volume

        points = np.concatenate(
            [surface_points, surface_points_dilated, center.reshape(-1, 3)],
            axis=0,
        )
        points_target = np.concatenate(
            [
                surface_points_change,
                surface_points_dilated,
                center.reshape(-1, 3),
            ],
            axis=0,
        )
        if self.use_grid_controls:
            grid_64 = get_grid_points()
            grid_64 = [point for point in grid_64 if label_dilated[tuple(point)] != 1]
            points = np.concatenate([points, grid_64], axis=0)
            points_target = np.concatenate([points_target, grid_64], axis=0)

        return points, points_target, amp
