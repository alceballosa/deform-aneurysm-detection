"""
Consistency tests for crop2 module.

Tests that verify the DetectionCropper class in the refactored version
maintains consistent behavior with reasonable expectations.

Note: The reference version uses InstanceCrop2 with different API,
so these tests focus on validating the current implementation.
"""

import sys
import importlib
import numpy as np

from .conftest import (
    CURRENT_PROJECT_ROOT,
    assert_arrays_close,
)


def import_crop2_module(project_root):
    """
    Import crop2 module from specified project.

    Args:
        project_root: Path to project root

    Returns:
        Imported module
    """
    sys.path.insert(0, str(project_root))
    try:
        # Clear cached modules
        modules_to_clear = [
            'src.dataset.crop2',
            'src.dataset.crop',
            'src.dataset',
            'src',
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        module = importlib.import_module('src.dataset.crop2')
        return module

    finally:
        sys.path.remove(str(project_root))


class TestDetectionCropperInit:
    """Test DetectionCropper initialization in current version."""

    def test_init_basic_params(self):
        """Test initialization with basic parameters."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [64, 96, 96]
        spacing = [1.0, 1.0, 1.0]
        overlap = [16, 32, 32]

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            spacing=spacing,
            overlap=overlap,
        )

        # Verify attributes are set correctly
        assert cropper.crop_size == crop_size
        assert cropper.spacing == spacing
        assert cropper.overlap == overlap
        assert isinstance(cropper.tp_ratio, float)
        assert isinstance(cropper.sample_num, int)

    def test_init_with_augmentation_params(self):
        """Test initialization with augmentation parameters."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [64, 96, 96]
        rand_trans = [5, 5, 5]
        rand_rot = [10, 10, 0]
        rand_space = [0.9, 1.1]

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            rand_trans=rand_trans,
            rand_rot=rand_rot,
            rand_space=rand_space,
        )

        # Verify augmentation parameters
        assert_arrays_close(
            cropper.rand_trans,
            np.array(rand_trans),
            name="rand_trans",
        )
        assert_arrays_close(
            cropper.rand_rot,
            np.array(rand_rot),
            name="rand_rot",
        )
        assert_arrays_close(
            cropper.rand_space,
            np.array(rand_space),
            name="rand_space",
        )

    def test_init_with_none_augmentations(self):
        """Test initialization with None augmentation parameters."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [64, 96, 96]

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        # All augmentations should be None
        assert cropper.rand_trans is None
        assert cropper.rand_rot is None
        assert cropper.rand_space is None


class TestDetectionCropperBehavior:
    """Test DetectionCropper behavior in current version."""

    def create_mock_data(self, seed=42):
        """Create mock data for testing."""
        np.random.seed(seed)

        # Create mock image
        image = np.random.randn(64, 128, 128).astype(np.float32)

        # Create mock annotations (lesion locations, radii, and classes)
        all_loc = np.array([
            [32, 64, 64],  # z, y, x
            [48, 80, 80],
        ], dtype=np.float32)

        all_rad = np.array([
            [5, 5, 5],  # d, h, w
            [3, 3, 3],
        ], dtype=np.float32)

        all_cls = np.array([0, 0], dtype=np.int8)  # Both aneurysms

        image_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        return {
            "image": image,
            "all_loc": all_loc,
            "all_rad": all_rad,
            "all_cls": all_cls,
            "image_spacing": image_spacing,
            "scan_id": "test_scan.nii.gz",
        }

    def test_cropper_output_structure(self):
        """Test that cropper produces expected output structure."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]
        sample_num = 2

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            sample_num=sample_num,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        # Create mock data
        data = self.create_mock_data()

        # Get crops
        np.random.seed(42)
        samples = cropper(data)

        # Verify number of samples
        assert len(samples) == sample_num

        # Verify structure of samples
        for sample in samples:
            assert "image" in sample
            assert isinstance(sample["image"], np.ndarray)
            # Shape is (C, H, W, D) with channel dimension
            assert sample["image"].ndim == 4
            assert sample["image"].shape[0] == 1  # Single channel

    def test_cropper_deterministic_with_seed(self):
        """Test that cropper produces deterministic results with fixed seed."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]
        sample_num = 2

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            sample_num=sample_num,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        data = self.create_mock_data()

        # Run twice with same seed
        np.random.seed(42)
        samples_1 = cropper(data)

        np.random.seed(42)
        samples_2 = cropper(data)

        # Results with same seed should be identical
        assert len(samples_1) == len(samples_2)
        for s1, s2 in zip(samples_1, samples_2):
            assert_arrays_close(s1["image"], s2["image"], name="image")

    def test_cropper_sample_number(self):
        """Test that cropper produces correct number of samples."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]

        for sample_num in [1, 2, 4]:
            cropper = current_mod.DetectionCropper(
                crop_size=crop_size,
                sample_num=sample_num,
            )

            data = self.create_mock_data()

            np.random.seed(42)
            samples = cropper(data)

            assert len(samples) == sample_num

    def test_cropper_output_shape(self):
        """Test that cropper produces correct output shapes."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]
        sample_num = 2

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            sample_num=sample_num,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        data = self.create_mock_data()

        np.random.seed(42)
        samples = cropper(data)

        # Check that all samples have correct dimensions
        # Shape is (C, H, W, D) with channel dimension
        for sample in samples:
            assert sample["image"].ndim == 4
            assert sample["image"].shape[0] == 1  # Single channel

    def test_cropper_with_mask(self):
        """Test cropper with vessel mask."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]
        sample_num = 1

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            sample_num=sample_num,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        # Create mock data with vessel mask
        data = self.create_mock_data()
        np.random.seed(42)
        data["mask"] = (np.random.rand(64, 128, 128) > 0.9).astype(np.float32)

        np.random.seed(42)
        samples = cropper(data)

        # Should produce mask in output
        assert "mask" in samples[0]

        # Mask should have same dimensionality as image
        assert samples[0]["mask"].ndim == 4
        assert samples[0]["mask"].shape[0] == 1  # Single channel


class TestCropperEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cropper_with_no_annotations(self):
        """Test cropper behavior with empty annotations."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        crop_size = [32, 48, 48]
        sample_num = 2

        cropper = current_mod.DetectionCropper(
            crop_size=crop_size,
            sample_num=sample_num,
            instance_crop=False,  # Disable instance cropping
        )

        # Create data with no annotations
        np.random.seed(42)
        data = {
            "image": np.random.randn(64, 128, 128).astype(np.float32),
            "all_loc": np.array([], dtype=np.float32).reshape(0, 3),
            "all_rad": np.array([], dtype=np.float32).reshape(0, 3),
            "all_cls": np.array([], dtype=np.int8),
            "image_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "scan_id": "test_scan.nii.gz",
        }

        np.random.seed(42)
        samples = cropper(data)

        # Should still produce samples (random crops)
        assert len(samples) == sample_num

    def test_cropper_tp_ratio(self):
        """Test that TP ratio parameter is set consistently."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        for tp_ratio in [0.0, 0.5, 0.7, 1.0]:
            cropper = current_mod.DetectionCropper(
                crop_size=[32, 48, 48],
                tp_ratio=tp_ratio,
            )

            assert cropper.tp_ratio == tp_ratio

    def test_cropper_callable(self):
        """Test that cropper is callable and produces output."""
        current_mod = import_crop2_module(CURRENT_PROJECT_ROOT)

        cropper = current_mod.DetectionCropper(
            crop_size=[32, 48, 48],
            sample_num=1,
        )

        # Verify it's callable
        assert callable(cropper)

        # Create minimal data
        np.random.seed(42)
        data = {
            "image": np.random.randn(64, 128, 128).astype(np.float32),
            "all_loc": np.array([[32, 64, 64]], dtype=np.float32),
            "all_rad": np.array([[5, 5, 5]], dtype=np.float32),
            "all_cls": np.array([0], dtype=np.int8),
            "image_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "scan_id": "test_scan.nii.gz",
        }

        # Should run without error
        samples = cropper(data)
        assert len(samples) > 0
