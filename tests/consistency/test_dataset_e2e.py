"""
End-to-end consistency tests for dataset module.

These tests compare actual outputs between current and reference implementations
by running the same operations on both versions and comparing results.
"""

import sys
import importlib
import tempfile
import numpy as np
import SimpleITK as sitk

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
    assert_arrays_close,
)


def import_dataset_mapper_module(project_root):
    """Import dataset_mapper from specified project."""
    sys.path.insert(0, str(project_root))
    try:
        modules_to_clear = [
            'src.dataset.dataset_mapper',
            'src.dataset.crop2',
            'src.dataset.crop',
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


class TestDataLoadingEndToEnd:
    """End-to-end tests comparing data loading pipeline."""

    def create_test_scan(self):
        """Create a test NIfTI scan."""
        # Create reproducible test data
        np.random.seed(42)
        data = np.random.randn(32, 64, 64).astype(np.float32) * 100 + 50

        image = sitk.GetImageFromArray(data)
        image.SetSpacing((1.0, 1.0, 1.0))

        tmp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        sitk.WriteImage(image, tmp_file.name)

        return tmp_file.name, data

    def test_zscore_normalization_consistency(self):
        """Test that z-score normalization produces identical results."""
        from yacs.config import CfgNode as CN

        # Create minimal config
        cfg = CN()
        cfg.DATA = CN()
        cfg.DATA.PATCH_SIZE = [32, 64, 64]
        cfg.DATA.SPACING = [1.0, 1.0, 1.0]
        cfg.DATA.OVERLAP = [8, 16, 16]
        cfg.DATA.WINDOW = [-100, 300]
        cfg.DATA.NORM_TYPE = "zscore"
        cfg.DATA.CROPPING_AUG = CN()
        cfg.DATA.CROPPING_AUG.SPACING = [1.0, 1.0]
        cfg.DATA.CROPPING_AUG.ROTATION = [0, 0, 0]
        cfg.DATA.CROPPING_AUG.TRANSLATION = [0, 0, 0]
        cfg.DATA.CROPPING_AUG.TP_RATIO = 0.7
        cfg.DATA.CROPPING_AUG.BLANK_SIDE = 0
        cfg.DATA.CROPPING_AUG.PADDED_REORIENT = False
        cfg.DATA.CROPPING_AUG.TRANSFORM_RAD = False
        cfg.MODEL = CN()
        cfg.MODEL.USE_VESSEL_INFO = "no"
        cfg.MODEL.USE_CVS_INFO = "no"
        cfg.SOLVER = CN()
        cfg.SOLVER.SAMPLES_PER_SCAN = 1

        # Create test scan
        test_file, original_data = self.create_test_scan()

        # Create dataset dict
        dataset_dict = {
            "file_name": test_file,
            "scan_id": "test_scan.nii.gz",
            "annotations": np.array([[32, 32, 16, 3, 3, 3, "aneurysm"]]),
        }

        # Load with current version
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        current_mapper = current_mod.CTADatasetMapper(cfg, mode="train")
        current_data = current_mapper.load_data(dataset_dict)

        # Load with reference version
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)
        reference_mapper = reference_mod.CTADatasetMapper(cfg, mode="train")
        reference_data = reference_mapper.load_data(dataset_dict)

        # Compare normalized images
        assert_arrays_close(
            current_data["image"],
            reference_data["image"],
            rtol=1e-6,
            atol=1e-8,
            name="normalized_image"
        )

        # Compare annotations
        assert_arrays_close(
            current_data["all_loc"],
            reference_data["all_loc"],
            name="all_loc"
        )
        assert_arrays_close(
            current_data["all_rad"],
            reference_data["all_rad"],
            name="all_rad"
        )

    def test_window_normalization_consistency(self):
        """Test that window-based normalization is identical."""
        from yacs.config import CfgNode as CN

        cfg = CN()
        cfg.DATA = CN()
        cfg.DATA.PATCH_SIZE = [32, 64, 64]
        cfg.DATA.SPACING = [1.0, 1.0, 1.0]
        cfg.DATA.OVERLAP = [8, 16, 16]
        cfg.DATA.WINDOW = [-100, 300]
        cfg.DATA.NORM_TYPE = "zscore_clamp"
        cfg.DATA.CROPPING_AUG = CN()
        cfg.DATA.CROPPING_AUG.SPACING = [1.0, 1.0]
        cfg.DATA.CROPPING_AUG.ROTATION = [0, 0, 0]
        cfg.DATA.CROPPING_AUG.TRANSLATION = [0, 0, 0]
        cfg.DATA.CROPPING_AUG.TP_RATIO = 0.7
        cfg.DATA.CROPPING_AUG.BLANK_SIDE = 0
        cfg.DATA.CROPPING_AUG.PADDED_REORIENT = False
        cfg.DATA.CROPPING_AUG.TRANSFORM_RAD = False
        cfg.MODEL = CN()
        cfg.MODEL.USE_VESSEL_INFO = "no"
        cfg.MODEL.USE_CVS_INFO = "no"
        cfg.SOLVER = CN()
        cfg.SOLVER.SAMPLES_PER_SCAN = 1

        test_file, _ = self.create_test_scan()

        dataset_dict = {
            "file_name": test_file,
            "scan_id": "test_scan.nii.gz",
            "annotations": np.array([[32, 32, 16, 3, 3, 3, "aneurysm"]]),
        }

        # Load with both versions
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        current_mapper = current_mod.CTADatasetMapper(cfg, mode="train")
        current_data = current_mapper.load_data(dataset_dict)

        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)
        reference_mapper = reference_mod.CTADatasetMapper(cfg, mode="train")
        reference_data = reference_mapper.load_data(dataset_dict)

        # Compare
        assert_arrays_close(
            current_data["image"],
            reference_data["image"],
            rtol=1e-6,
            atol=1e-8,
            name="clamped_normalized_image"
        )


class TestDatasetDictConsistency:
    """Test that dataset dict generation is consistent."""

    def test_path_resolution_consistency(self):
        """Test that path resolution produces same results."""
        current_mod = import_dataset_mapper_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_mapper_module(REFERENCE_PROJECT_ROOT)

        # Import dataset_dict module
        sys.path.insert(0, str(CURRENT_PROJECT_ROOT))
        current_dict_mod = importlib.import_module('src.dataset.dataset_dict')
        sys.path.remove(str(CURRENT_PROJECT_ROOT))

        sys.path.insert(0, str(REFERENCE_PROJECT_ROOT))
        if 'src.dataset.dataset_dict' in sys.modules:
            del sys.modules['src.dataset.dataset_dict']
        if 'src.dataset' in sys.modules:
            del sys.modules['src.dataset']
        reference_dict_mod = importlib.import_module('src.dataset.dataset_dict')
        sys.path.remove(str(REFERENCE_PROJECT_ROOT))

        # Test various paths
        test_paths = [
            "/tmp/scan.nii.gz",
            "./data/scan.nii.gz",
            "../data/scan.nii.gz",
        ]

        for test_path in test_paths:
            current_resolved = current_dict_mod.resolve_path(test_path)
            reference_resolved = reference_dict_mod.resolve_path(test_path)

            # Both should produce absolute paths
            assert current_resolved == reference_resolved, (
                f"Path resolution differs for {test_path}: "
                f"{current_resolved} vs {reference_resolved}"
            )


class TestGetDatasetDictsConsistency:
    """Test dataset catalog retrieval consistency."""

    def test_get_dataset_dicts_output_structure(self):
        """Test that get_dataset_dicts produces identical structures."""
        # Import dataloader from both versions
        sys.path.insert(0, str(CURRENT_PROJECT_ROOT))
        current_dataloader = importlib.import_module('src.dataset.dataloader')
        sys.path.remove(str(CURRENT_PROJECT_ROOT))

        sys.path.insert(0, str(REFERENCE_PROJECT_ROOT))
        modules_to_clear = ['src.dataset.dataloader', 'src.dataset', 'src']
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        reference_dataloader = importlib.import_module('src.dataset.dataloader')
        sys.path.remove(str(REFERENCE_PROJECT_ROOT))

        # Create mock dataset
        mock_dicts = [
            {"scan_id": "scan_001", "file_name": "/tmp/scan_001.nii.gz"},
            {"scan_id": "scan_002", "file_name": "/tmp/scan_002.nii.gz"},
        ]

        try:
            from detectron2.data import DatasetCatalog

            # Register mock dataset
            if "consistency_test" in DatasetCatalog.list():
                DatasetCatalog.remove("consistency_test")

            DatasetCatalog.register("consistency_test", lambda: mock_dicts.copy())

            # Get from both versions
            current_result = current_dataloader.get_dataset_dicts("consistency_test")
            reference_result = reference_dataloader.get_dataset_dicts("consistency_test")

            # Compare
            assert len(current_result) == len(reference_result)
            for curr, ref in zip(current_result, reference_result):
                assert curr == ref, f"Dataset dict mismatch: {curr} vs {ref}"

        finally:
            if "consistency_test" in DatasetCatalog.list():
                DatasetCatalog.remove("consistency_test")
