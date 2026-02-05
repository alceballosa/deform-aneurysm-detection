"""
Instance-based cropping and augmentation for 3D medical images.

This module provides functionality for extracting training patches from 3D medical
images with spatial augmentations. It supports random rotation, translation, spacing
variation, and intelligent sampling strategies to balance positive and negative examples.
"""

from __future__ import division, print_function

import logging
import random

import numpy as np
import SimpleITK as sitk


class DetectionCropper:
    """
    Extract training patches from 3D medical images with spatial augmentation.

    This class randomly crops patches from volumetric images with various augmentations
    including rotation, translation, and spacing variations. It implements intelligent
    sampling to ensure a balanced mix of patches containing lesions (positive) and
    background patches (negative).

    Key features:
    - Random sampling from regular grid positions and near lesion centers
    - Spatial augmentations: rotation, translation, spacing variation
    - Configurable true positive sampling ratio
    - Support for vessel and CVS mask propagation
    - Physical coordinate space transformations using SimpleITK

    Attributes:
        crop_size: Patch dimensions in voxels [D, H, W]
        tp_ratio: Target ratio of patches containing at least one lesion
        sample_num: Number of patches to extract per scan
        blank_side: Border width to ignore for lesion labels
        instance_crop: Whether to include patches centered on lesions
        overlap: Overlap between adjacent patches [D, H, W]
        spacing: Target voxel spacing [z, y, x]
        rand_trans: Random translation range in voxels (or None)
        rand_rot: Random rotation range in degrees (or None)
        rand_space: Random spacing range (or None)
        sample_cls: List of lesion class IDs to sample
        base_spacing: Base spacing for coordinate transformations
        padded_reorient: Whether to use padded reorientation
    """

    def __init__(
        self,
        crop_size,
        rand_trans=None,
        rand_rot=None,
        rand_space=None,
        instance_crop=True,
        spacing=[1.0, 1.0, 1.0],
        overlap=[16, 32, 32],
        tp_ratio=0.7,
        sample_num=2,
        blank_side=0,
        padded_reorient=False,
        sample_cls=[0, 1],
    ):
        """
        Initialize the cropping function with augmentation parameters.

        Args:
            crop_size: Patch size in voxels [D, H, W]
            rand_trans: Random translation range in voxels (default: None for no translation)
            rand_rot: Random rotation range in degrees [D, H, W] (default: None for no rotation)
            rand_space: Random spacing range [min, max] (default: None for fixed spacing)
            instance_crop: If True, sample patches near lesion centers (default: True)
            spacing: Output patch spacing [z, y, x] (default: [1.0, 1.0, 1.0])
            overlap: Overlap between patches in voxels [D, H, W] (default: [16, 32, 32])
            tp_ratio: Target ratio of positive patches (default: 0.7)
            sample_num: Number of patches to extract per scan (default: 2)
            blank_side: Border width to ignore for labels in pixels (default: 0)
            padded_reorient: Use padded reorientation for transforms (default: False)
            sample_cls: List of lesion class IDs to consider as positive (default: [0])
        """
        self.crop_size = crop_size
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop
        self.overlap = overlap
        self.spacing = spacing

        if rand_trans is None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        if rand_rot is None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)

        if rand_space is None:
            self.rand_space = None
        else:
            self.rand_space = np.array(rand_space)

        self.sample_cls = sample_cls
        self.base_spacing = spacing
        self.padded_reorient = padded_reorient
        assert isinstance(self.crop_size, (list, tuple))

    def __call__(self, sample):
        """
        Extract and augment patches from a 3D medical image.

        This method:
        1. Determines patch centers (mix of regular grid and lesion-centered)
        2. Applies spatial augmentations (rotation, translation, spacing)
        3. Extracts patches and transforms annotations to patch coordinates
        4. Returns list of augmented patches with corresponding annotations

        Args:
            sample: Dictionary containing:
                - "image": 3D numpy array (D, H, W)
                - "scan_id": Unique scan identifier
                - "all_loc": Lesion locations in voxel coordinates (N, 3)
                - "all_rad": Lesion radii (N, 3)
                - "all_cls": Lesion class labels (N,)
                - "image_spacing": Voxel spacing (3,)
                - "vessel_edt": Optional vessel segmentation mask
                - "cvs_mask": Optional CVS mask

        Returns:
            list: List of patch dictionaries, each containing:
                - "scan_id": Scan identifier
                - "image": Patch array with shape (1, D, H, W)
                - "ctr": Lesion centers in patch coordinates
                - "rad": Lesion radii in patch coordinates
                - "cls": Lesion class labels
                - "vessel_edt": Optional vessel mask patch
                - "cvs_mask": Optional CVS mask patch
                - "volume": Optional total vessel volume in patch
        """
        # Extract metadata
        scan_id = sample["scan_id"]
        all_loc = sample["all_loc"]
        all_rad = sample["all_rad"]
        all_cls = sample["all_cls"]

        image_spacing = sample["image_spacing"]


        instance_loc = all_loc[
            np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype="bool")
        ]

        # Build dictionary of arrays to process with their interpolation types
        arrays_to_process = {
            "image": {
                "data": sample["image"].astype("float32"),
                "interp": sitk.sitkLinear,
            }
        }

        # Add optional arrays if present
        if "vessel_edt" in sample:
            arrays_to_process["vessel_edt"] = {
                "data": sample["vessel_edt"],
                "interp": sitk.sitkNearestNeighbor,
            }
        if "cvs_mask" in sample:
            arrays_to_process["cvs_mask"] = {
                "data": sample["cvs_mask"],
                "interp": sitk.sitkNearestNeighbor,
            }

        # Convert all arrays to SimpleITK format
        arrays_itk = {}
        for key, array_info in arrays_to_process.items():
            arrays_itk[key] = sitk.GetImageFromArray(array_info["data"])

        # Create shadow image for coordinate transformations
        shape = arrays_to_process["image"]["data"].shape
        shadow = np.zeros(shape)
        shadow_itk = sitk.GetImageFromArray(shadow)

        re_spacing = np.array(self.spacing) / np.array(self.base_spacing)
        crop_size = np.array(self.crop_size) * re_spacing
        overlap = self.overlap * re_spacing

        # Determine number of positive vs negative samples
        if self.sample_num > 1:
            if len(instance_loc) > 0:
                num_pos_samples = int(np.ceil(self.sample_num * self.tp_ratio))
            else:
                num_pos_samples = 0  # no positive samples available
        else:
            # For single sample, randomly decide based on tp_ratio
            num_pos_samples = np.random.choice(
                [0, 1], p=[1 - self.tp_ratio, self.tp_ratio]
            )
        num_rand_samples = self.sample_num - num_pos_samples

        # Get patch centers at regular grid positions
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
        # todo: ensure the random centers are not outside the brain
        rand_indices = np.random.choice(
            len(crop_centers), size=num_rand_samples, replace=False
        )
        rand_centers = np.array(crop_centers)[rand_indices]

        # Get patch centers near lesions
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
        # Initialize crop storage for each array type
        array_crops = {key: [] for key in arrays_to_process.keys()}
        image_spacing_crops = []

        # Extract and reorient patches for all arrays
        for i in range(len(all_centers)):
            matrix = matrix_crops[i]
            space = space_crops[i]

            # Process each array type uniformly
            for key, array_info in arrays_to_process.items():
                array_itk_crop = reorient(
                    arrays_itk[key],
                    matrix,
                    crop_size,
                    spacing=list(space),
                    interp1=array_info["interp"],
                    padded_reorient=self.padded_reorient,
                )
                array_crop = sitk.GetArrayFromImage(array_itk_crop)
                array_crops[key].append(np.expand_dims(array_crop, axis=0))

            image_spacing_crops.append(space)



        # Build output samples
        samples = []
        num_patches = len(all_centers)
        for i in range(num_patches):
            ctr = all_loc_crops[i]
            rad = all_rad_crops[i]
            cla = all_cls_crops[i]

            scale_spacing = image_spacing_crops[i]
            real_space = scale_spacing

            if len(rad) > 0:
                rad = rad / real_space  # convert to pixel coordinates

            # Build sample dictionary with all cropped arrays
            patch_sample = {
                "scan_id": scan_id,
                "ctr": ctr,
                "rad": rad,
                "cls": cla,
            }

            # Add all cropped arrays to the sample
            for key in array_crops.keys():
                patch_sample[key] = array_crops[key][i]

            # Add special vessel volume metric if mask is present
            if "vessel_edt" in array_crops:
                patch_sample["volume"] = array_crops["vessel_edt"][i].sum()
            samples.append(patch_sample)
            #print("Avail", all_cls, "  |  Chosen:", patch_sample["cls"])
            
        #print("---------------")

        return samples

    def make_patch(
        self, C, re_spacing, crop_size, shadow_itk, all_cls, all_loc, all_rad
    ):
        """
        Create a single patch with augmentation and transform annotations.

        This method applies spatial augmentations (translation, rotation, spacing)
        and determines which lesions fall within the patch boundaries.

        Args:
            C: Patch center coordinates in voxels (3,)
            re_spacing: Relative spacing multiplier (3,)
            crop_size: Patch size in voxels (3,)
            shadow_itk: SimpleITK image for coordinate transformations
            all_cls: All lesion class labels (N,)
            all_loc: All lesion locations (N, 3)
            all_rad: All lesion radii (N, 3)

        Returns:
            tuple: (matrix, space, all_loc_crop, all_rad_crop, all_cls_crop, n_tp) where:
                - matrix: 4x3 transformation matrix for ITK reorientation
                - space: Final spacing after augmentation
                - all_loc_crop: Lesion centers in patch coordinates
                - all_rad_crop: Lesion radii for patches
                - all_cls_crop: Lesion classes in patch
                - n_tp: Number of target positive instances in patch
        """
        # Apply random translation
        if self.rand_trans is not None:
            C = (
                C
                + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=3)
                * re_spacing
            )

        # Define patch corners
        O = C - np.array(crop_size) / 2
        Z = O + np.array([crop_size[0] - 1, 0, 0])
        Y = O + np.array([0, crop_size[1] - 1, 0])
        X = O + np.array([0, 0, crop_size[2] - 1])
        matrix = np.array([O, X, Y, Z])

        # Apply random rotation
        if self.rand_rot is not None:
            matrix = rand_rot_coord(
                matrix,
                [-self.rand_rot[0], self.rand_rot[0]],
                [-self.rand_rot[1], self.rand_rot[1]],
                [-self.rand_rot[2], self.rand_rot[2]],
                rot_center=C,
                p=0.8,
            )

        # Apply random spacing variation
        if (self.rand_space is not None) and (random.random() < 0.8):
            space = (
                np.random.uniform(self.rand_space[0], self.rand_space[1], size=3)
                * re_spacing
            )
        else:
            space = re_spacing
        matrix = matrix[:, ::-1]  # Convert to ITK axis order

        # Create ITK image with transformation for coordinate mapping
        image_itk_crop = reorient(
            shadow_itk,
            matrix,
            crop_size,
            spacing=list(space),
            interp1=sitk.sitkNearestNeighbor,
            padded_reorient=self.padded_reorient,
        )

        # Transform lesion annotations to patch coordinates
        all_loc_crop = [
            image_itk_crop.TransformPhysicalPointToContinuousIndex(c.tolist()[::-1])[
                ::-1
            ]
            for c in all_loc
        ]
        all_loc_crop = np.array(all_loc_crop)

        # Determine which lesions are within patch bounds
        in_idx = []
        for j in range(all_loc_crop.shape[0]):
            if (all_loc_crop[j] <= np.array(image_itk_crop.GetSize()[::-1])).all() and (
                all_loc_crop[j] >= np.zeros([3])
            ).all():
                in_idx.append(True)
            else:
                in_idx.append(False)
        in_idx = np.array(in_idx)

        if in_idx.size > 0:
            all_loc_crop = all_loc_crop[in_idx]
            all_rad_crop = all_rad[in_idx]
            all_cls_crop = all_cls[in_idx]
        else:
            all_loc_crop = np.array([]).reshape(-1, 3)
            all_rad_crop = np.array([])
            all_cls_crop = np.array([])

        # Count number of target class instances in patch
        if all_cls.shape[0] == 0:
            n_tp = 0
        else:
            n_tp = np.sum(
                [all_cls[in_idx] == cls for cls in self.sample_cls],
                axis=0,
                dtype="bool",
            ).sum()

        return matrix, space, all_loc_crop, all_rad_crop, all_cls_crop, n_tp


# Backward compatibility alias
InstanceCrop = DetectionCropper


def rotate_vecs_3d(vec, angle, axis):
    """
    Rotate 3D vectors around a specified axis.

    Args:
        vec: Array of vectors to rotate (N, 3)
        angle: Rotation angle in degrees
        axis: Tuple (axis1, axis2) specifying the rotation plane

    Returns:
        np.ndarray: Rotated vectors with same shape as input
    """
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[
        ::, axis[1]
    ] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[
        ::, axis[1]
    ] * np.cos(rad)
    return rotated_vec


def apply_transformation_coord(coord, transform_param_list, rot_center):
    """
    Apply a sequence of rotation transformations to coordinates.

    Args:
        coord: Coordinate array to transform (N, 3)
        transform_param_list: List of (angle, axes) tuples for each rotation
        rot_center: Center point for rotations (3,)

    Returns:
        np.ndarray: Transformed coordinates
    """
    for angle, axes in transform_param_list:
        org = coord - rot_center
        new = rotate_vecs_3d(org, angle, axes)
        coord = new + rot_center

    return coord


def rand_rot_coord(coord, angle_range_d, angle_range_h, angle_range_w, rot_center, p):
    """
    Apply random rotation augmentation to coordinates.

    Randomly rotates coordinates around depth, height, and width axes
    with specified angle ranges. Each rotation is applied with probability p.

    Args:
        coord: Coordinates to rotate (N, 3)
        angle_range_d: [min, max] angle range for depth axis rotation (degrees)
        angle_range_h: [min, max] angle range for height axis rotation (degrees)
        angle_range_w: [min, max] angle range for width axis rotation (degrees)
        rot_center: Center point for rotations (3,)
        p: Probability of applying each rotation

    Returns:
        np.ndarray: Rotated coordinates
    """
    transform_param_list = []

    if (angle_range_d[1] - angle_range_d[0] > 0) and (random.random() < p):
        angle_d = np.random.uniform(angle_range_d[0], angle_range_d[1])
        transform_param_list.append([angle_d, (-2, -1)])
    if (angle_range_h[1] - angle_range_h[0] > 0) and (random.random() < p):
        angle_h = np.random.uniform(angle_range_h[0], angle_range_h[1])
        transform_param_list.append([angle_h, (-3, -1)])
    if (angle_range_w[1] - angle_range_w[0] > 0) and (random.random() < p):
        angle_w = np.random.uniform(angle_range_w[0], angle_range_w[1])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord


def reorient(
    itk_img,
    mark_matrix,
    crop_size,
    spacing=[1.0, 1.0, 1.0],
    interp1=sitk.sitkLinear,
    padded_reorient=False,
):
    """
    Reorient and resample a SimpleITK image based on physical mark points.

    This function performs arbitrary 3D reorientation and resampling by defining
    a new coordinate system using four physical mark points (origin and three
    axis endpoints).

    Args:
        itk_img: SimpleITK image to reorient
        mark_matrix: 4x3 array of physical mark points in ITK coordinate order:
                     [origin, x_end, y_end, z_end]
        crop_size: Output size in voxels [D, H, W]
        spacing: Output voxel spacing in ITK order [x, y, z] (default: [1.0, 1.0, 1.0])
        interp1: SimpleITK interpolator (default: sitk.sitkLinear)
        padded_reorient: If True, use ceil for size calculation; if False, use exact
                         crop_size (default: False)

    Returns:
        SimpleITK.Image: Reoriented and resampled image
    """
    spacing = spacing[::-1]  # Convert to ITK order (x, y, z)
    origin, x_mark, y_mark, z_mark = (
        np.array(mark_matrix[0]),
        np.array(mark_matrix[1]),
        np.array(mark_matrix[2]),
        np.array(mark_matrix[3]),
    )

    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetInterpolator(interp1)
    filter_resample.SetOutputSpacing(spacing)

    # Set origin
    origin_reorient = mark_matrix[0]

    # Set direction (column-wise basis vectors)
    x_base = (x_mark - origin) / np.linalg.norm(x_mark - origin)
    y_base = (y_mark - origin) / np.linalg.norm(y_mark - origin)
    z_base = (z_mark - origin) / np.linalg.norm(z_mark - origin)
    direction_reorient = (
        np.stack([x_base, y_base, z_base]).transpose().reshape(-1).tolist()
    )

    # Set output size
    x, y, z = (
        np.linalg.norm(x_mark - origin) / spacing[0],
        np.linalg.norm(y_mark - origin) / spacing[1],
        np.linalg.norm(z_mark - origin) / spacing[2],
    )

    if padded_reorient:
        size_reorient = (
            int(np.ceil(x + 0.5)),
            int(np.ceil(y + 0.5)),
            int(np.ceil(z + 0.5)),
        )
    else:
        size_reorient = (int(crop_size[0]), int(crop_size[1]), int(crop_size[2]))

    filter_resample.SetOutputOrigin(origin_reorient)
    filter_resample.SetOutputDirection(direction_reorient)
    filter_resample.SetSize(size_reorient)
    filter_resample.SetOutputPixelType(itk_img.GetPixelID())
    itk_out = filter_resample.Execute(itk_img)

    return itk_out
