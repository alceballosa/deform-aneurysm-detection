"""
Consistency tests for dataset_mapper module.

Tests that verify the CTADatasetMapper class produces identical outputs
in both versions for loading and preprocessing medical images.
"""

import sys
import importlib
import numpy as np
import pytest
import tempfile
import SimpleITK as sitk

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
    assert_arrays_close,
)


def import_dataset_mapper_module(project_root):
    """
    Import dataset_mapper module from specified project.

    Args:
        project_root: Path to project root

    Returns:
        Imported module
    """
    sys.path.insert(0, str(project_root))
    try:
        # Clear cached modules
        modules_to_clear = [
            'src.dataset.dataset_mapper',
            'src.dataset.crop2',
            'src.dataset.split_comb',
            'src.dataset',
            'src.transform',
            'src',
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        module = importlib.import_module('src.dataset.dataset_mapper')
        return module
    finally:
        sys.path.remove(str(project_root))


@pytest.fixture
def mock_cfg():
    """Create a mock configuration for dataset mapper."""
    from yacs.config import CfgNode as CN

    cfg = CN()

    # Data configuration
    cfg.DATA = CN()
    cfg.DATA.PATCH_SIZE = [64, 96, 96]
    cfg.DATA.SPACING = [1.0, 1.0, 1.0]
    cfg.DATA.OVERLAP = [16, 32, 32]
    cfg.DATA.WINDOW = [-100, 300]
    cfg.DATA.NORM_TYPE = "zscore"

    # Cropping augmentation
    cfg.DATA.CROPPING_AUG = CN()
    cfg.DATA.CROPPING_AUG.SPACING = [0.9, 1.1]
    cfg.DATA.CROPPING_AUG.ROTATION = [10, 10, 0]
    cfg.DATA.CROPPING_AUG.TRANSLATION = [5, 5, 5]
    cfg.DATA.CROPPING_AUG.TP_RATIO = 0.7
    cfg.DATA.CROPPING_AUG.BLANK_SIDE = 0
    cfg.DATA.CROPPING_AUG.PADDED_REORIENT = False
    cfg.DATA.CROPPING_AUG.TRANSFORM_RAD = False

    # Model configuration
    cfg.MODEL = CN()
    cfg.MODEL.USE_VESSEL_INFO = "no"
    cfg.MODEL.USE_CVS_INFO = "no"

    # Solver configuration
    cfg.SOLVER = CN()
    cfg.SOLVER.SAMPLES_PER_SCAN = 2

    return cfg


@pytest.fixture
def create_test_nifti():
    """Fixture to create test NIfTI files."""
    def _create(shape=(64, 128, 128), spacing=(1.0, 1.0, 1.0), seed=42):
        """Create a test NIfTI file and return the path."""
        np.random.seed(seed)
        data = np.random.randn(*shape).astype(np.float32) * 100 + 50

        image = sitk.GetImageFromArray(data)
        image.SetSpacing(spacing[::-1])  # ITK uses x, y, z order

        # Create temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        sitk.WriteImage(image, tmp_file.name)

        return tmp_file.name, data, spacing

    return _create


class TestDatasetMapperConstants:
    """Test that constants are identical between versions."""

    def test_lesion_labels_consistency(self):
        """Test that LESION_LABELS are identical."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper
        reference_mapper = reference_mod.CTADatasetMapper

        assert current_mapper.LESION_LABELS == reference_mapper.LESION_LABELS

    def test_lesion_ids_consistency(self):
        """Test that LESION_IDS are identical."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper
        reference_mapper = reference_mod.CTADatasetMapper

        assert current_mapper.LESION_IDS == reference_mapper.LESION_IDS

    def test_augmentation_constants_consistency(self):
        """Test that augmentation constants exist in current version."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)

        # These constants were added in the refactored version
        # Just verify they exist in the current version
        assert hasattr(current_mod, 'AUGMENTATION_DISABLE_THRESHOLD')
        assert hasattr(current_mod, 'FLIP_PROBABILITY')
        assert hasattr(current_mod, 'TRANSPOSE_PROBABILITY')
        assert hasattr(current_mod, 'POSITIVE_CROP_RATIO')

        # Verify reasonable values
        assert 0 < current_mod.FLIP_PROBABILITY <= 1.0
        assert 0 < current_mod.TRANSPOSE_PROBABILITY <= 1.0
        assert 0 < current_mod.POSITIVE_CROP_RATIO <= 1.0


class TestCropConfigConsistency:
    """Test crop configuration loading consistency."""

    def test_load_crop_cfg_basic(self, mock_cfg):
        """Test basic crop configuration loading."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        current_cfg = current_mapper._load_crop_cfg()
        reference_cfg = reference_mapper._load_crop_cfg()

        # Compare all configuration parameters
        assert current_cfg.keys() == reference_cfg.keys()
        assert current_cfg["crop_size"] == reference_cfg["crop_size"]
        assert current_cfg["spacing"] == reference_cfg["spacing"]
        assert current_cfg["overlap"] == reference_cfg["overlap"]
        assert current_cfg["tp_ratio"] == reference_cfg["tp_ratio"]
        assert current_cfg["sample_num"] == reference_cfg["sample_num"]
        assert current_cfg["blank_side"] == reference_cfg["blank_side"]
        assert current_cfg["padded_reorient"] == reference_cfg["padded_reorient"]

    def test_load_crop_cfg_disabled_augmentations(self, mock_cfg):
        """Test that augmentations are disabled when ranges are negligible."""
        # Set negligible augmentation ranges
        mock_cfg.DATA.CROPPING_AUG.SPACING = [1.0, 1.0001]
        mock_cfg.DATA.CROPPING_AUG.ROTATION = [0.0, 0.0, 0.0]
        mock_cfg.DATA.CROPPING_AUG.TRANSLATION = [0.0, 0.0, 0.0]

        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        current_cfg = current_mapper._load_crop_cfg()
        reference_cfg = reference_mapper._load_crop_cfg()

        # All augmentations should be None
        assert current_cfg["rand_space"] is None
        assert reference_cfg["rand_space"] is None
        assert current_cfg["rand_rot"] is None
        assert reference_cfg["rand_rot"] is None
        assert current_cfg["rand_trans"] is None
        assert reference_cfg["rand_trans"] is None


class TestNormalizationConsistency:
    """Test data normalization consistency."""

    def test_normalize_window_based(self, mock_cfg):
        """Test window-based normalization."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        # Create test data
        np.random.seed(42)
        test_data = np.random.randn(20, 30, 30).astype(np.float32) * 200

        # Normalize with both versions
        current_result = current_mapper.normalize(test_data.copy())
        reference_result = reference_mapper.normalize(test_data.copy())

        # Compare results
        assert_arrays_close(current_result, reference_result, name="normalized_data")

    def test_normalize_clipping(self, mock_cfg):
        """Test that normalization clips values consistently."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        # Create test data with extreme values
        test_data = np.array([-1000, -100, 0, 100, 300, 1000]).astype(np.float32)

        current_result = current_mapper.normalize(test_data.copy())
        reference_result = reference_mapper.normalize(test_data.copy())

        # Results should be in [0, 1] range
        assert current_result.min() >= 0.0
        assert current_result.max() <= 1.0
        assert reference_result.min() >= 0.0
        assert reference_result.max() <= 1.0

        # Results should be identical
        assert_arrays_close(current_result, reference_result, name="clipped_data")


class TestLoadDataConsistency:
    """Test data loading consistency."""

    def test_load_data_zscore_normalization(self, mock_cfg, create_test_nifti):
        """Test loading data with z-score normalization."""
        mock_cfg.DATA.NORM_TYPE = "zscore"

        # Create test NIfTI file
        test_file, test_data, spacing = create_test_nifti()

        dataset_dict = {
            "file_name": test_file,
            "scan_id": "test_scan.nii.gz",
            "annotations": np.array([[64, 64, 32, 5, 5, 5, "aneurysm"]]),
        }

        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        # Load data with both versions
        current_result = current_mapper.load_data(dataset_dict)
        reference_result = reference_mapper.load_data(dataset_dict)

        # Compare results
        assert current_result.keys() == reference_result.keys()
        assert current_result["scan_id"] == reference_result["scan_id"]

        # Compare image data (should be z-score normalized)
        assert_arrays_close(
            current_result["image"],
            reference_result["image"],
            name="image",
            rtol=1e-5,
        )

        # Compare spacing
        assert_arrays_close(
            np.array(current_result["image_spacing"]),
            np.array(reference_result["image_spacing"]),
            name="spacing",
        )

        # Compare annotations
        assert_arrays_close(
            current_result["all_loc"],
            reference_result["all_loc"],
            name="all_loc",
        )
        assert_arrays_close(
            current_result["all_rad"],
            reference_result["all_rad"],
            name="all_rad",
        )
        assert_arrays_close(
            current_result["all_cls"],
            reference_result["all_cls"],
            name="all_cls",
        )

    def test_load_data_zscore_clamp_normalization(self, mock_cfg, create_test_nifti):
        """Test loading data with clamped z-score normalization."""
        mock_cfg.DATA.NORM_TYPE = "zscore_clamp"

        # Create test NIfTI file
        test_file, test_data, spacing = create_test_nifti()

        dataset_dict = {
            "file_name": test_file,
            "scan_id": "test_scan.nii.gz",
            "annotations": np.array([[64, 64, 32, 5, 5, 5, "aneurysm"]]),
        }

        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        current_mapper = current_mod.CTADatasetMapper(mock_cfg, mode="train")
        reference_mapper = reference_mod.CTADatasetMapper(mock_cfg, mode="train")

        # Load data with both versions
        current_result = current_mapper.load_data(dataset_dict)
        reference_result = reference_mapper.load_data(dataset_dict)

        # Compare image data
        assert_arrays_close(
            current_result["image"],
            reference_result["image"],
            name="clamped_image",
            rtol=1e-5,
        )


class TestMaybeReadFromRAM:
    """Test RAM cache reading consistency."""

    def test_maybe_read_from_ram_fallback(self, create_test_nifti):
        """Test that RAM cache reading falls back to disk consistently."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        # Create test file
        test_file, test_data, spacing = create_test_nifti()

        # Read with both versions (should fall back to disk)
        current_image = current_mod.maybe_read_from_ram(test_file)
        reference_image = reference_mod.maybe_read_from_ram(test_file)

        # Convert to arrays and compare
        current_array = sitk.GetArrayFromImage(current_image)
        reference_array = sitk.GetArrayFromImage(reference_image)

        assert_arrays_close(current_array, reference_array, name="ram_image")

        # Compare spacing
        assert current_image.GetSpacing() == reference_image.GetSpacing()
