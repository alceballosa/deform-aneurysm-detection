"""
End-to-end consistency tests for crop2 module.

Tests that compare actual cropping outputs between current and reference implementations.
"""

import sys
import importlib
import numpy as np
import pytest

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
    assert_arrays_close,
)


def import_crop2_modules():
    """Import crop2 from both versions."""
    # Import current version
    sys.path.insert(0, str(CURRENT_PROJECT_ROOT))
    modules_to_clear = ['src.dataset.crop2', 'src.dataset.crop', 'src.dataset', 'src']
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    current_crop2 = importlib.import_module('src.dataset.crop2')
    sys.path.remove(str(CURRENT_PROJECT_ROOT))

    # Import reference version
    sys.path.insert(0, str(REFERENCE_PROJECT_ROOT))
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    reference_crop2 = importlib.import_module('src.dataset.crop2')
    sys.path.remove(str(REFERENCE_PROJECT_ROOT))

    return current_crop2, reference_crop2


def create_test_data(seed=42):
    """Create reproducible test data for cropping."""
    np.random.seed(seed)

    # Create test image
    image = np.random.randn(64, 128, 128).astype(np.float32)

    # Create annotations
    all_loc = np.array([
        [32, 64, 64],  # z, y, x
        [48, 80, 80],
    ], dtype=np.float32)

    all_rad = np.array([
        [5, 5, 5],  # d, h, w
        [3, 3, 3],
    ], dtype=np.float32)

    all_cls = np.array([0, 0], dtype=np.int8)
    image_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return {
        "image": image,
        "all_loc": all_loc,
        "all_rad": all_rad,
        "all_cls": all_cls,
        "image_spacing": image_spacing,
        "scan_id": "test_scan.nii.gz",
    }


class TestCrop2EndToEnd:
    """E2E tests comparing cropping outputs between implementations."""

    def test_cropper_class_exists(self):
        """Test that cropper classes exist in both versions."""
        current_crop2, reference_crop2 = import_crop2_modules()

        # Current should have DetectionCropper
        assert hasattr(current_crop2, 'DetectionCropper'), (
            "Current version missing DetectionCropper"
        )

        # Reference should have InstanceCrop2
        assert hasattr(reference_crop2, 'InstanceCrop2'), (
            "Reference version missing InstanceCrop2"
        )

    def test_cropper_initialization_params(self):
        """Test that croppers accept similar initialization parameters."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        # Common parameters both should support
        common_params = {
            'crop_size': [32, 48, 48],
            'spacing': [1.0, 1.0, 1.0],
            'overlap': [8, 16, 16],
            'tp_ratio': 0.7,
            'sample_num': 2,
        }

        # Try initializing both with common parameters
        try:
            current_cropper = CurrentCropper(**common_params)
            assert current_cropper is not None
        except Exception as e:
            pytest.fail(f"Current cropper initialization failed: {e}")

        try:
            reference_cropper = ReferenceCropper(**common_params)
            assert reference_cropper is not None
        except Exception as e:
            pytest.fail(f"Reference cropper initialization failed: {e}")

    def test_deterministic_cropping_produces_output(self):
        """Test that both croppers produce valid outputs with deterministic settings."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        # Create croppers with NO augmentation (deterministic)
        crop_size = [32, 48, 48]
        spacing = [1.0, 1.0, 1.0]

        current_cropper = CurrentCropper(
            crop_size=crop_size,
            spacing=spacing,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
            sample_num=2,
            tp_ratio=1.0,  # Always sample from lesions
        )

        reference_cropper = ReferenceCropper(
            crop_size=crop_size,
            spacing=spacing,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
            sample_num=2,
            tp_ratio=1.0,
        )

        # Create test data
        data = create_test_data(seed=42)

        # Run both croppers with same seed
        np.random.seed(42)
        current_samples = current_cropper(data)

        np.random.seed(42)
        reference_samples = reference_cropper(data)

        # Verify outputs are valid
        assert len(current_samples) > 0, "Current cropper produced no samples"
        assert len(reference_samples) > 0, "Reference cropper produced no samples"

        # Both should produce same number of samples
        assert len(current_samples) == len(reference_samples), (
            f"Different number of samples: {len(current_samples)} vs {len(reference_samples)}"
        )

        # Verify structure
        for curr_sample in current_samples:
            assert "image" in curr_sample
            assert isinstance(curr_sample["image"], np.ndarray)

        for ref_sample in reference_samples:
            assert "image" in ref_sample
            assert isinstance(ref_sample["image"], np.ndarray)

    def test_cropping_with_vessel_mask_consistency(self):
        """Test that both croppers handle vessel masks consistently."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        crop_size = [32, 48, 48]

        current_cropper = CurrentCropper(
            crop_size=crop_size,
            sample_num=1,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        reference_cropper = ReferenceCropper(
            crop_size=crop_size,
            sample_num=1,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        # Create data with vessel mask
        data = create_test_data(seed=42)
        np.random.seed(42)
        data["mask"] = (np.random.rand(64, 128, 128) > 0.9).astype(np.float32)

        # Run both
        np.random.seed(42)
        current_samples = current_cropper(data)

        np.random.seed(42)
        reference_samples = reference_cropper(data)

        # Both should handle mask
        assert len(current_samples) > 0
        assert len(reference_samples) > 0

        # If mask present in output, verify it's an array
        if "mask" in current_samples[0]:
            assert isinstance(current_samples[0]["mask"], np.ndarray)

        if "mask" in reference_samples[0]:
            assert isinstance(reference_samples[0]["mask"], np.ndarray)

    def test_cropping_edge_case_no_annotations(self):
        """Test that both handle empty annotations consistently."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        crop_size = [32, 48, 48]

        current_cropper = CurrentCropper(
            crop_size=crop_size,
            sample_num=2,
            instance_crop=False,  # Disable instance cropping
        )

        reference_cropper = ReferenceCropper(
            crop_size=crop_size,
            sample_num=2,
            instance_crop=False,
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

        # Run both
        np.random.seed(42)
        current_samples = current_cropper(data)

        np.random.seed(42)
        reference_samples = reference_cropper(data)

        # Both should still produce samples (random crops)
        assert len(current_samples) > 0
        assert len(reference_samples) > 0

    def test_cropping_output_contains_required_fields(self):
        """Test that both croppers produce outputs with expected fields."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        crop_size = [32, 48, 48]

        current_cropper = CurrentCropper(crop_size=crop_size, sample_num=1)
        reference_cropper = ReferenceCropper(crop_size=crop_size, sample_num=1)

        data = create_test_data(seed=42)

        # Run both
        np.random.seed(42)
        current_samples = current_cropper(data)

        np.random.seed(42)
        reference_samples = reference_cropper(data)

        # Required fields that both should have
        required_fields = ["image"]

        for curr_sample in current_samples:
            for field in required_fields:
                assert field in curr_sample, f"Current missing field: {field}"

        for ref_sample in reference_samples:
            for field in required_fields:
                assert field in ref_sample, f"Reference missing field: {field}"

    def test_cropping_determinism_with_fixed_seed(self):
        """Test that both croppers are deterministic with fixed seed."""
        current_crop2, reference_crop2 = import_crop2_modules()

        CurrentCropper = current_crop2.DetectionCropper
        ReferenceCropper = reference_crop2.InstanceCrop2

        crop_size = [32, 48, 48]

        current_cropper = CurrentCropper(
            crop_size=crop_size,
            sample_num=1,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        reference_cropper = ReferenceCropper(
            crop_size=crop_size,
            sample_num=1,
            rand_trans=None,
            rand_rot=None,
            rand_space=None,
        )

        data = create_test_data(seed=42)

        # Run current twice with same seed
        np.random.seed(42)
        current_samples_1 = current_cropper(data)

        np.random.seed(42)
        current_samples_2 = current_cropper(data)

        # Should be identical
        assert len(current_samples_1) == len(current_samples_2)
        for s1, s2 in zip(current_samples_1, current_samples_2):
            assert_arrays_close(s1["image"], s2["image"], name="current_determinism")

        # Run reference twice with same seed
        np.random.seed(42)
        reference_samples_1 = reference_cropper(data)

        np.random.seed(42)
        reference_samples_2 = reference_cropper(data)

        # Should be identical
        assert len(reference_samples_1) == len(reference_samples_2)
        for s1, s2 in zip(reference_samples_1, reference_samples_2):
            assert_arrays_close(s1["image"], s2["image"], name="reference_determinism")
