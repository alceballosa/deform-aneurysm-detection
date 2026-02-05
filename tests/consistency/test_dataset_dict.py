"""
Consistency tests for dataset_dict module.

Tests that verify the CTADatasetFunction and path resolution utilities
produce identical outputs in both versions.
"""

import sys
import importlib
import numpy as np
import pytest
from pathlib import Path

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
    assert_dicts_close,
)


@pytest.fixture
def mock_cfg():
    """Create a mock configuration object for testing."""
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.CUSTOM = CN()
    cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
    cfg.CUSTOM.DEBUG = False
    cfg.CUSTOM.DEBUG_DATASET_SIZE = 2

    cfg.DATA = CN()
    cfg.DATA.DIR = CN()
    cfg.DATA.DIR.TRAIN = CN()
    cfg.DATA.DIR.TRAIN.SCAN_DIR = "/tmp/test_scans"
    cfg.DATA.DIR.TRAIN.LABEL_DIR = "/tmp/test_labels"
    cfg.DATA.DIR.TRAIN.VESSEL_DIR = "/tmp/test_vessels"
    cfg.DATA.DIR.TRAIN.CVS_DIR = "/tmp/test_cvs"
    cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = "/tmp/annotations.csv"

    cfg.DATA.DIR.VAL = CN()
    cfg.DATA.DIR.VAL.SCAN_DIR = "/tmp/test_scans_val"
    cfg.DATA.DIR.VAL.LABEL_DIR = ""
    cfg.DATA.DIR.VAL.VESSEL_DIR = "/tmp/test_vessels_val"
    cfg.DATA.DIR.VAL.CVS_DIR = "/tmp/test_cvs_val"

    cfg.MODEL = CN()
    cfg.MODEL.USE_VESSEL_INFO = "edt"
    cfg.MODEL.USE_CVS_INFO = "no"

    cfg.DATASETS = CN()
    cfg.DATASETS.TRAIN = ["cta_train"]
    cfg.DATASETS.TEST = ["cta_val"]

    return cfg


def import_dataset_dict_module(project_root):
    """
    Import dataset_dict module from specified project.

    Args:
        project_root: Path to project root

    Returns:
        Imported module
    """
    sys.path.insert(0, str(project_root))
    try:
        if 'src.dataset.dataset_dict' in sys.modules:
            del sys.modules['src.dataset.dataset_dict']
        if 'src.dataset' in sys.modules:
            del sys.modules['src.dataset']
        if 'src' in sys.modules:
            del sys.modules['src']

        module = importlib.import_module('src.dataset.dataset_dict')
        return module
    finally:
        sys.path.remove(str(project_root))


class TestResolvePathConsistency:
    """Test path resolution utility consistency."""

    def test_resolve_path_absolute(self):
        """Test that absolute paths are resolved consistently."""
        current_mod = import_dataset_dict_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_dict_module(REFERENCE_PROJECT_ROOT)

        test_path = "/tmp/test/scan.nii.gz"

        current_result = current_mod.resolve_path(test_path)
        reference_result = reference_mod.resolve_path(test_path)

        assert current_result == reference_result

    def test_resolve_path_relative(self):
        """Test that relative paths are resolved consistently."""
        current_mod = import_dataset_dict_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_dict_module(REFERENCE_PROJECT_ROOT)

        test_path = "./data/scan.nii.gz"

        current_result = current_mod.resolve_path(test_path)
        reference_result = reference_mod.resolve_path(test_path)

        # Both should resolve to absolute paths
        assert Path(current_result).is_absolute()
        assert Path(reference_result).is_absolute()


class TestCTADatasetFunctionConsistency:
    """Test CTADatasetFunction consistency between versions."""

    def test_dataset_function_structure(self, mock_cfg, tmp_path):
        """Test that dataset function produces consistent structure."""
        # Create minimal test data
        scan_dir = tmp_path / "scans"
        scan_dir.mkdir()
        (scan_dir / "scan_001.nii.gz").touch()
        (scan_dir / "scan_002.nii.gz").touch()

        vessel_dir = tmp_path / "vessels"
        vessel_dir.mkdir()
        (vessel_dir / "scan_001.nii.gz").touch()
        (vessel_dir / "scan_002.nii.gz").touch()

        # Create mock annotation file
        import pandas as pd
        annotations = pd.DataFrame({
            "seriesuid": ["scan_001.nii.gz", "scan_002.nii.gz"],
            "coordX": [64.0, 80.0],
            "coordY": [64.0, 80.0],
            "coordZ": [32.0, 40.0],
            "w": [5.0, 3.0],
            "h": [5.0, 3.0],
            "d": [5.0, 3.0],
            "lesion": ["aneurysm", "aneurysm"],
        })
        annot_file = tmp_path / "annotations.csv"
        annotations.to_csv(annot_file, index=False)

        # Update config with test paths
        mock_cfg.DATA.DIR.TRAIN.SCAN_DIR = str(scan_dir)
        mock_cfg.DATA.DIR.TRAIN.VESSEL_DIR = str(vessel_dir)
        mock_cfg.DATA.DIR.TRAIN.CVS_DIR = str(tmp_path / "cvs")
        mock_cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = str(annot_file)
        mock_cfg.DATA.DIR.TRAIN.LABEL_DIR = ""

        # Import both versions
        current_mod = import_dataset_dict_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_dict_module(REFERENCE_PROJECT_ROOT)

        # Create dataset functions
        current_fn = current_mod.CTADatasetFunction(mock_cfg, mode="train")
        reference_fn = reference_mod.CTADatasetFunction(mock_cfg, mode="train")

        # Get dataset dicts
        current_dicts = current_fn()
        reference_dicts = reference_fn()

        # Compare lengths
        assert len(current_dicts) == len(reference_dicts), (
            f"Dataset length mismatch: {len(current_dicts)} vs {len(reference_dicts)}"
        )

        # Compare each dictionary
        for i, (curr, ref) in enumerate(zip(current_dicts, reference_dicts)):
            assert curr["scan_id"] == ref["scan_id"], (
                f"Scan ID mismatch at index {i}"
            )

            # Check that required keys exist
            required_keys = ["scan_id", "file_name", "annotations"]
            for key in required_keys:
                assert key in curr, f"Missing key {key} in current version"
                assert key in ref, f"Missing key {key} in reference version"

            # Check annotations shape
            if curr["annotations"] is not None and ref["annotations"] is not None:
                assert curr["annotations"].shape == ref["annotations"].shape, (
                    f"Annotations shape mismatch at index {i}"
                )

    def test_val_mode_consistency(self, mock_cfg, tmp_path):
        """Test validation mode produces consistent outputs."""
        # Create minimal test data
        scan_dir = tmp_path / "scans_val"
        scan_dir.mkdir()
        (scan_dir / "val_001.nii.gz").touch()

        vessel_dir = tmp_path / "vessels_val"
        vessel_dir.mkdir()
        (vessel_dir / "val_001.nii.gz").touch()

        # Update config
        mock_cfg.DATA.DIR.VAL.SCAN_DIR = str(scan_dir)
        mock_cfg.DATA.DIR.VAL.VESSEL_DIR = str(vessel_dir)
        mock_cfg.DATA.DIR.VAL.LABEL_DIR = ""

        # Import both versions
        current_mod = import_dataset_dict_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_dict_module(REFERENCE_PROJECT_ROOT)

        # Create dataset functions
        current_fn = current_mod.CTADatasetFunction(mock_cfg, mode="val")
        reference_fn = reference_mod.CTADatasetFunction(mock_cfg, mode="val")

        # Get dataset dicts
        current_dicts = current_fn()
        reference_dicts = reference_fn()

        # Compare
        assert len(current_dicts) == len(reference_dicts)

        for curr, ref in zip(current_dicts, reference_dicts):
            assert curr["scan_id"] == ref["scan_id"]
            # In val mode, annotations should be None
            assert curr["annotations"] is None
            assert ref["annotations"] is None

    def test_debug_mode_limits_dataset(self, mock_cfg, tmp_path):
        """Test that debug mode limits dataset size consistently."""
        # Create test data with more scans than debug limit
        scan_dir = tmp_path / "scans"
        scan_dir.mkdir()
        for i in range(10):
            (scan_dir / f"scan_{i:03d}.nii.gz").touch()

        vessel_dir = tmp_path / "vessels"
        vessel_dir.mkdir()
        for i in range(10):
            (vessel_dir / f"scan_{i:03d}.nii.gz").touch()

        # Create mock annotations
        import pandas as pd
        annotations = pd.DataFrame({
            "seriesuid": [f"scan_{i:03d}.nii.gz" for i in range(10)],
            "coordX": [64.0] * 10,
            "coordY": [64.0] * 10,
            "coordZ": [32.0] * 10,
            "w": [5.0] * 10,
            "h": [5.0] * 10,
            "d": [5.0] * 10,
            "lesion": ["aneurysm"] * 10,
        })
        annot_file = tmp_path / "annotations.csv"
        annotations.to_csv(annot_file, index=False)

        # Update config
        mock_cfg.DATA.DIR.TRAIN.SCAN_DIR = str(scan_dir)
        mock_cfg.DATA.DIR.TRAIN.VESSEL_DIR = str(vessel_dir)
        mock_cfg.DATA.DIR.TRAIN.CVS_DIR = str(tmp_path / "cvs")
        mock_cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = str(annot_file)
        mock_cfg.DATA.DIR.TRAIN.LABEL_DIR = ""
        mock_cfg.CUSTOM.DEBUG = True
        mock_cfg.CUSTOM.DEBUG_DATASET_SIZE = 3

        # Import both versions
        current_mod = import_dataset_dict_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataset_dict_module(REFERENCE_PROJECT_ROOT)

        # Create dataset functions
        current_fn = current_mod.CTADatasetFunction(mock_cfg, mode="train")
        reference_fn = reference_mod.CTADatasetFunction(mock_cfg, mode="train")

        # Get dataset dicts
        current_dicts = current_fn()
        reference_dicts = reference_fn()

        # Both should limit to DEBUG_DATASET_SIZE
        assert len(current_dicts) == 3
        assert len(reference_dicts) == 3
        assert len(current_dicts) == len(reference_dicts)
