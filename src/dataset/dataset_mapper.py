"""
Dataset mapper for CTA (CT Angiography) scan processing.

This module provides the CTADatasetMapper class for loading and preprocessing
medical images with Detectron2 integration for aneurysm detection tasks.
"""

import copy
import os

import edt
import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from detectron2.utils.registry import Registry

from src import transform

from .crop2 import DetectionCropper
from .split_comb import SplitComb

DATA_MAPPER_REGISTRY = Registry("DATA_MAPPER")

# Constants for augmentation configuration
AUGMENTATION_DISABLE_THRESHOLD = 1e-3  # Threshold for disabling augmentations
FLIP_PROBABILITY = 0.5  # Probability for random flipping
TRANSPOSE_PROBABILITY = 0.5  # Probability for random transposing
POSITIVE_CROP_RATIO = 0.9  # Ratio for positive region cropping


@DATA_MAPPER_REGISTRY.register()
class CTADatasetMapper:
    """
    Dataset mapper for CTA (CT Angiography) scans used with Detectron2.

    This class handles loading, preprocessing, and augmenting 3D medical images
    for aneurysm detection. It supports both training and validation modes with
    optional vessel and CVS (Circle of Willis) mask information.

    Attributes:
        LESION_LABELS: List of valid lesion type labels
        LESION_IDS: Dictionary mapping lesion labels to integer IDs
    """

    LESION_LABELS = ["aneurysm", "non_aneurysm"]
    LESION_IDS = {
        "aneurysm": 0,
        "non_aneurysm": 1,
    }

    def __init__(self, cfg, mode):
        """
        Initialize the dataset mapper.

        Args:
            cfg: Configuration object containing data processing parameters including:
                - DATA.PATCH_SIZE: Size of crops for training
                - DATA.CROPPING_AUG: Cropping augmentation parameters
                - DATA.SPACING: Target voxel spacing
                - DATA.OVERLAP: Overlap for patch extraction
                - MODEL.USE_VESSEL_INFO: Whether to use vessel masks
                - MODEL.USE_CVS_INFO: Whether to use CVS masks
            mode: Operating mode, either "train" or "val"
                - "train": Applies augmentations and random cropping
                - "val": Uses split-combine for full volume processing
        """
        assert mode in ["train", "val"]
        self.cfg = cfg
        if mode == "train":
            self.augmentations = self.build_transforms()
            self.crop_fn = self.build_crop_fn()
        else:
            self.split_comb = self.build_split_comb()
        self.mode = mode

    def build_crop_fn(self):
        """
        Build the cropping function for extracting training patches.

        Returns:
            InstanceCrop2: Configured crop function that extracts patches from volumes
        """
        cfg = self._load_crop_cfg()
        return DetectionCropper(**cfg)

    def _load_crop_cfg(self):
        """
        Load and validate cropping configuration parameters.

        Extracts cropping parameters from config and disables augmentations
        (spacing, rotation, translation) if their ranges are negligible.

        Returns:
            dict: Configuration dictionary for InstanceCrop2 containing:
                - crop_size: Patch dimensions
                - rand_trans: Random translation range (or None if disabled)
                - rand_rot: Random rotation range (or None if disabled)
                - rand_space: Random spacing range (or None if disabled)
                - spacing: Target voxel spacing
                - overlap: Patch overlap ratio
                - tp_ratio: True positive sampling ratio
                - sample_num: Number of samples per scan
                - blank_side: Blank side padding
                - padded_reorient: Whether to use padded reorientation
        """
        cfg = self.cfg

        random_space = cfg.DATA.CROPPING_AUG.SPACING
        if (random_space[1] - random_space[0]) < AUGMENTATION_DISABLE_THRESHOLD:
            random_space = None

        random_rotation = cfg.DATA.CROPPING_AUG.ROTATION
        if max(random_rotation) < AUGMENTATION_DISABLE_THRESHOLD:
            random_rotation = None

        random_translation = cfg.DATA.CROPPING_AUG.TRANSLATION
        if max(random_translation) < AUGMENTATION_DISABLE_THRESHOLD:
            random_translation = None

        return dict(
            crop_size=cfg.DATA.PATCH_SIZE,
            rand_trans=random_translation,
            rand_rot=random_rotation,
            rand_space=random_space,
            spacing=cfg.DATA.SPACING,
            overlap=cfg.DATA.OVERLAP,
            tp_ratio=cfg.DATA.CROPPING_AUG.TP_RATIO,
            sample_num=cfg.SOLVER.SAMPLES_PER_SCAN,
            blank_side=cfg.DATA.CROPPING_AUG.BLANK_SIDE,
            padded_reorient=cfg.DATA.CROPPING_AUG.PADDED_REORIENT,
        )

    def build_split_comb(self):
        """
        Build the split-combine function for validation mode.

        Used to split large volumes into overlapping patches for inference
        and later combine predictions back into full volume.

        Returns:
            SplitComb: Configured split-combine function
        """
        cfg = self.cfg
        return SplitComb(
            crop_size=cfg.DATA.PATCH_SIZE,
            overlap=cfg.DATA.OVERLAP,
            pad_value=-1,  # normalized minimum
        )

    def build_transforms(self):
        """
        Build the data augmentation pipeline for training.

        Transforms include random flipping, transposing, padding, and cropping.
        Vessel and CVS masks are automatically handled if present in the sample.

        Returns:
            torchvision.transforms.Compose: Composed transform pipeline
        """
        crop_size = self.cfg.DATA.PATCH_SIZE
        transform_list_train = [
            transform.RandomFlip(
                flip_depth=True,
                flip_height=True,
                flip_width=True,
                p=FLIP_PROBABILITY,
            ),
            transform.RandomTranspose(
                trans_xy=True,
                trans_zx=False,
                trans_zy=False,
                p=TRANSPOSE_PROBABILITY,
                transform_rad=self.cfg.DATA.CROPPING_AUG.TRANSFORM_RAD,
            ),
            transform.Pad(output_size=crop_size),
            transform.RandomCrop(
                output_size=crop_size, pos_ratio=POSITIVE_CROP_RATIO
            ),
            transform.CoordToAnnot(),
        ]
        return torchvision.transforms.Compose(transform_list_train)

    def __call__(self, dataset_dict):
        """
        Process a single dataset item for training or validation.

        Args:
            dataset_dict: Dictionary containing scan metadata and file paths.
                Required keys: "file_name", "scan_id"
                Optional keys: "annotations", "vessel_file_name", "cvs_file_name"

        Returns:
            dict: Processed dataset dictionary with:
                - For training: "samples" list of augmented patches
                - For validation: "image", "image_spacing", and optional masks
        """
        import time

        pid = os.getpid()
        scan_id = dataset_dict.get("scan_id", "?")
        t0 = time.monotonic()
        _log_step = lambda tag: print(
            f"[MAPPER pid={pid}] {tag} scan={scan_id} "
            f"dt={time.monotonic() - t0:.1f}s",
            flush=True,
        )

        _log_step("START")
        dataset_dict = copy.deepcopy(dataset_dict)
        _log_step("deepcopy done")
        data = self.load_data(dataset_dict)
        _log_step("load_data done")

        if self.mode == "train":
            samples = self.crop_fn(data)
            _log_step(f"crop done ({len(samples)} samples)")
            random_samples = []
            for sample in samples:
                if self.augmentations:
                    sample = self.augmentations(sample)
                random_samples.append(sample)
            _log_step("augmentations done")

            dataset_dict["samples"] = random_samples
        else:
            dataset_dict["image"] = torch.tensor(data["image"], device="cpu")
            dataset_dict["image_spacing"] = data["image_spacing"]

            if self.cfg.MODEL.USE_VESSEL_INFO != "no":
                dataset_dict["vessel_edt"] = torch.tensor(data["vessel_edt"], device="cpu")

            if self.cfg.MODEL.USE_CVS_INFO != "no":
                dataset_dict["cvs_mask"] = torch.tensor(data["cvs_mask"], device="cpu")
        _log_step("END")
        return dataset_dict

    def load_data(self, dataset_dict):
        """
        Load and preprocess medical image data from disk or RAM.

        Loads the main CTA image and optionally vessel and CVS masks.
        Applies normalization based on configuration settings.

        Args:
            dataset_dict: Dictionary with file paths and metadata including:
                - "file_name": Path to main CTA scan
                - "scan_id": Unique scan identifier
                - "annotations": Training annotations (train mode only)
                - "vessel_file_name": Path to vessel mask (optional)
                - "cvs_file_name": Path to CVS mask (optional)

        Returns:
            dict: Processed data containing:
                - "image": Normalized 3D image array (z,y,x)
                - "image_spacing": Voxel spacing (z,y,x)
                - "scan_id": Scan identifier
                - "all_loc": Lesion locations in voxel coords (train only)
                - "all_rad": Lesion radii (train only)
                - "all_cls": Lesion class labels (train only)
                - "vessel_edt": Vessel mask (if configured)
                - "cvs_mask": CVS mask (if configured)
        """
        import time

        pid = os.getpid()
        scan_id = dataset_dict.get("scan_id", "?")
        t0 = time.monotonic()
        _log = lambda tag: print(
            f"[LOAD pid={pid}] {tag} scan={scan_id} "
            f"dt={time.monotonic() - t0:.1f}s",
            flush=True,
        )

        outputs = {}

        _log("reading image")
        image = maybe_read_from_ram(dataset_dict["file_name"])
        _log("image read done")
        image_spacing = image.GetSpacing()[::-1]  # z, y, x
        image = sitk.GetArrayFromImage(image).astype("float32")  # z, y, x

        if self.cfg.DATA.NORM_TYPE == "base":
            image = self.normalize(image)
        elif self.cfg.DATA.NORM_TYPE == "zscore":
            mean_value = image.mean()
            std_value = image.std()
            image = (image - mean_value) / std_value
        elif self.cfg.DATA.NORM_TYPE == "zscore_clamp":
            image = np.clip(image, self.cfg.DATA.WINDOW[0], self.cfg.DATA.WINDOW[1])
            mean_value = image.mean()
            std_value = image.std()
            image = (image - mean_value) / std_value
        _log("normalization done")

        outputs["image"] = image
        outputs["image_spacing"] = image_spacing
        outputs["scan_id"] = dataset_dict["scan_id"]

        if self.mode == "train":
            csv_label = dataset_dict["annotations"]
            all_loc = csv_label[:, 0:3].astype("float32")  # x,y,z
            all_loc = all_loc[:, ::-1]  # convert to z,y,x
            all_rad = csv_label[:, 3:6].astype("float32")  # w,h,d
            all_rad = all_rad[:, ::-1]  # convert to d,h,w
            all_cls_text = csv_label[:, -1]  # lesion type
            all_cls = np.array(
                [self.LESION_IDS[label] for label in all_cls_text], dtype="int8"
            )

            outputs["all_loc"] = all_loc
            outputs["all_rad"] = all_rad
            outputs["all_cls"] = all_cls

        if self.cfg.MODEL.USE_VESSEL_INFO == "no":
            _log("done (no vessel)")
            return outputs

        _log("reading vessel EDT")
        vessel_header = maybe_read_from_ram(dataset_dict["vessel_file_name"])
        vessel = sitk.GetArrayFromImage(vessel_header).astype("float32")
        outputs["vessel_edt"] = vessel
        _log("vessel EDT done")

        if self.cfg.MODEL.USE_CVS_INFO != "no":
            _log("reading CVS mask")
            cvs_header = maybe_read_from_ram(dataset_dict["cvs_file_name"])
            cvs = sitk.GetArrayFromImage(cvs_header).astype("float32")
            outputs["cvs_mask"] = cvs
            _log("CVS mask done")

        _log("done")
        return outputs

    def normalize(self, data):
        """
        Normalize image data using window-based normalization to [-1, 1].

        Clips values to the configured window range and normalizes to [-1, 1].

        Args:
            data: Input image array

        Returns:
            np.ndarray: Normalized image in range [-1, 1]
        """
        min_value, max_value = self.cfg.DATA.WINDOW
        data = np.clip(data, min_value, max_value)
        data = (data - (min_value + max_value) / 2) / ((max_value - min_value) / 2)
        return data


def maybe_read_from_ram(file_name, _slow_threshold=30.0):
    """
    Attempt to read a medical image from RAM cache, falling back to disk.

    Tries to read the file from /dev/shm/ (RAM disk) for faster access.
    If the file doesn't exist in RAM or has a different size than the
    original, falls back to reading from the original file location.

    Args:
        file_name: Path to the original file on disk

    Returns:
        SimpleITK.Image: The loaded medical image

    Raises:
        RuntimeError: If the file cannot be read from either location
    """
    import time

    t0 = time.monotonic()
    new_folder = "/dev/shm/"
    sample_name = "/".join(file_name.split("/")[-3:])
    new_file_name = os.path.join(new_folder, sample_name)
    try:
        size_og = os.path.getsize(file_name)
        size_new = os.path.getsize(new_file_name)
        if size_og == size_new:
            img = sitk.ReadImage(new_file_name)
        else:
            print(
                f"Size mismatch: {file_name} ({size_og}) != "
                f"{new_file_name} ({size_new})"
            )
            img = sitk.ReadImage(file_name)
    except (FileNotFoundError, OSError):
        # File not in RAM cache, read from original location
        img = sitk.ReadImage(file_name)

    elapsed = time.monotonic() - t0
    if elapsed > _slow_threshold:
        print(
            f"[SLOW READ] {elapsed:.1f}s reading {file_name} "
            f"(worker pid={os.getpid()})",
            flush=True,
        )
    return img
