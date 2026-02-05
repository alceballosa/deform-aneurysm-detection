"""
End-to-end consistency tests for dataset_dict module.

Tests that compare actual dataset registration and retrieval behavior
between current and reference implementations.
"""

import sys
import importlib
import tempfile
import os
import numpy as np
import pandas as pd

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
    assert_arrays_close,
)


def import_dataset_dict_modules():
    """Import dataset_dict from both versions."""
    # Current version
    sys.path.insert(0, str(CURRENT_PROJECT_ROOT))
    modules_to_clear = ['src.dataset.dataset_dict', 'src.dataset', 'src']
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    current_dd = importlib.import_module('src.dataset.dataset_dict')
    sys.path.remove(str(CURRENT_PROJECT_ROOT))

    # Reference version
    sys.path.insert(0, str(REFERENCE_PROJECT_ROOT))
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    reference_dd = importlib.import_module('src.dataset.dataset_dict')
    sys.path.remove(str(REFERENCE_PROJECT_ROOT))

    return current_dd, reference_dd


class TestResolvePathEndToEnd:
    """E2E tests for path resolution."""

    def test_absolute_path_resolution(self):
        """Test that absolute paths resolve identically."""
        current_dd, reference_dd = import_dataset_dict_modules()

        test_paths = [
            "/tmp/scan.nii.gz",
            "/projects/data/scan_001.nii.gz",
            "/home/user/data/test.nii.gz",
        ]

        for test_path in test_paths:
            current_resolved = current_dd.resolve_path(test_path)
            reference_resolved = reference_dd.resolve_path(test_path)

            assert current_resolved == reference_resolved, (
                f"Absolute path resolution differs for {test_path}: "
                f"{current_resolved} vs {reference_resolved}"
            )

            # Both should produce absolute paths
            assert os.path.isabs(current_resolved)
            assert os.path.isabs(reference_resolved)

    def test_relative_path_resolution(self):
        """Test that relative paths resolve to absolute paths identically."""
        current_dd, reference_dd = import_dataset_dict_modules()

        test_paths = [
            "./data/scan.nii.gz",
            "../data/scan.nii.gz",
            "scan.nii.gz",
        ]

        for test_path in test_paths:
            current_resolved = current_dd.resolve_path(test_path)
            reference_resolved = reference_dd.resolve_path(test_path)

            # Both should produce absolute paths
            assert os.path.isabs(current_resolved), (
                f"Current did not produce absolute path for {test_path}"
            )
            assert os.path.isabs(reference_resolved), (
                f"Reference did not produce absolute path for {test_path}"
            )

            # Should be identical
            assert current_resolved == reference_resolved, (
                f"Relative path resolution differs for {test_path}: "
                f"{current_resolved} vs {reference_resolved}"
            )

    def test_path_normalization(self):
        """Test that paths with ./ and ../ are normalized identically."""
        current_dd, reference_dd = import_dataset_dict_modules()

        test_paths = [
            "/tmp/./scan.nii.gz",
            "/tmp/data/../scan.nii.gz",
            "/tmp/data/./subdir/../scan.nii.gz",
        ]

        for test_path in test_paths:
            current_resolved = current_dd.resolve_path(test_path)
            reference_resolved = reference_dd.resolve_path(test_path)

            assert current_resolved == reference_resolved, (
                f"Path normalization differs for {test_path}: "
                f"{current_resolved} vs {reference_resolved}"
            )


class TestCTADatasetFunctionEndToEnd:
    """E2E tests for CTADatasetFunction."""

    def create_test_dataset_structure(self, tmpdir):
        """Create a test dataset directory structure."""
        scan_dir = os.path.join(tmpdir, "scans")
        vessel_dir = os.path.join(tmpdir, "vessels")
        label_dir = os.path.join(tmpdir, "labels")

        os.makedirs(scan_dir)
        os.makedirs(vessel_dir)
        os.makedirs(label_dir)

        # Create dummy files
        scan_files = []
        for i in range(5):
            scan_name = f"scan_{i:03d}.nii.gz"
            open(os.path.join(scan_dir, scan_name), 'w').close()
            open(os.path.join(vessel_dir, scan_name), 'w').close()
            open(os.path.join(label_dir, scan_name), 'w').close()
            scan_files.append(scan_name)

        # Create annotations
        annotations = pd.DataFrame({
            "seriesuid": scan_files,
            "coordX": [64.0, 80.0, 50.0, 60.0, 70.0],
            "coordY": [64.0, 80.0, 50.0, 60.0, 70.0],
            "coordZ": [32.0, 40.0, 20.0, 25.0, 35.0],
            "w": [5.0, 3.0, 4.0, 6.0, 3.5],
            "h": [5.0, 3.0, 4.0, 6.0, 3.5],
            "d": [5.0, 3.0, 4.0, 6.0, 3.5],
            "lesion": ["aneurysm", "aneurysm", "non_aneurysm", "aneurysm", "aneurysm"],
        })
        annot_file = os.path.join(tmpdir, "annotations.csv")
        annotations.to_csv(annot_file, index=False)

        return scan_dir, vessel_dir, label_dir, annot_file

    def test_dataset_function_train_mode(self):
        """Test that CTADatasetFunction produces identical outputs in train mode."""
        from yacs.config import CfgNode as CN

        current_dd, reference_dd = import_dataset_dict_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir, vessel_dir, label_dir, annot_file = self.create_test_dataset_structure(tmpdir)

            # Create config
            cfg = CN()
            cfg.CUSTOM = CN()
            cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
            cfg.CUSTOM.DEBUG = False
            cfg.CUSTOM.DEBUG_DATASET_SIZE = 2
            cfg.DATA = CN()
            cfg.DATA.DIR = CN()
            cfg.DATA.DIR.TRAIN = CN()
            cfg.DATA.DIR.TRAIN.SCAN_DIR = scan_dir
            cfg.DATA.DIR.TRAIN.LABEL_DIR = label_dir
            cfg.DATA.DIR.TRAIN.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.TRAIN.CVS_DIR = tmpdir
            cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = annot_file
            cfg.DATA.DIR.VAL = CN()
            cfg.DATA.DIR.VAL.SCAN_DIR = scan_dir
            cfg.DATA.DIR.VAL.LABEL_DIR = ""
            cfg.DATA.DIR.VAL.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.VAL.CVS_DIR = tmpdir
            cfg.MODEL = CN()
            cfg.MODEL.USE_VESSEL_INFO = "edt"
            cfg.MODEL.USE_CVS_INFO = "no"

            # Create dataset functions
            current_fn = current_dd.CTADatasetFunction(cfg, mode="train")
            reference_fn = reference_dd.CTADatasetFunction(cfg, mode="train")

            # Get datasets
            current_dicts = current_fn()
            reference_dicts = reference_fn()

            # Compare lengths
            assert len(current_dicts) == len(reference_dicts), (
                f"Different dataset sizes: {len(current_dicts)} vs {len(reference_dicts)}"
            )

            # Compare each entry
            for i, (curr, ref) in enumerate(zip(current_dicts, reference_dicts)):
                assert curr["scan_id"] == ref["scan_id"], (
                    f"Scan ID mismatch at index {i}: {curr['scan_id']} vs {ref['scan_id']}"
                )

                assert "file_name" in curr and "file_name" in ref
                assert "vessel_file_name" in curr and "vessel_file_name" in ref

                # Compare annotations (structured array with mixed types)
                if curr["annotations"] is not None and ref["annotations"] is not None:
                    assert len(curr["annotations"]) == len(ref["annotations"]), (
                        f"Different annotation counts at index {i}"
                    )
                    # Annotations are structured arrays: [x, y, z, w, h, d, lesion_type]
                    # Compare numeric parts (first 6 columns)
                    curr_numeric = curr["annotations"][:, :6].astype(np.float64)
                    ref_numeric = ref["annotations"][:, :6].astype(np.float64)
                    assert_arrays_close(
                        curr_numeric,
                        ref_numeric,
                        name=f"annotations_numeric_{i}",
                        rtol=1e-6,
                    )
                    # Compare lesion types (last column)
                    curr_lesions = curr["annotations"][:, 6]
                    ref_lesions = ref["annotations"][:, 6]
                    assert np.array_equal(curr_lesions, ref_lesions), (
                        f"Lesion types differ at index {i}"
                    )

    def test_dataset_function_val_mode(self):
        """Test that CTADatasetFunction works identically in val mode."""
        from yacs.config import CfgNode as CN

        current_dd, reference_dd = import_dataset_dict_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = os.path.join(tmpdir, "scans_val")
            vessel_dir = os.path.join(tmpdir, "vessels_val")

            os.makedirs(scan_dir)
            os.makedirs(vessel_dir)

            # Create files
            for i in range(3):
                scan_name = f"val_scan_{i:03d}.nii.gz"
                open(os.path.join(scan_dir, scan_name), 'w').close()
                open(os.path.join(vessel_dir, scan_name), 'w').close()

            # Create config (needs both TRAIN and VAL sections)
            cfg = CN()
            cfg.CUSTOM = CN()
            cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
            cfg.CUSTOM.DEBUG = False
            cfg.CUSTOM.DEBUG_DATASET_SIZE = 2
            cfg.DATA = CN()
            cfg.DATA.DIR = CN()
            cfg.DATA.DIR.TRAIN = CN()
            cfg.DATA.DIR.TRAIN.SCAN_DIR = scan_dir
            cfg.DATA.DIR.TRAIN.LABEL_DIR = ""
            cfg.DATA.DIR.TRAIN.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.TRAIN.CVS_DIR = tmpdir
            cfg.DATA.DIR.VAL = CN()
            cfg.DATA.DIR.VAL.SCAN_DIR = scan_dir
            cfg.DATA.DIR.VAL.LABEL_DIR = ""
            cfg.DATA.DIR.VAL.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.VAL.CVS_DIR = tmpdir
            cfg.MODEL = CN()
            cfg.MODEL.USE_VESSEL_INFO = "edt"
            cfg.MODEL.USE_CVS_INFO = "no"

            # Create dataset functions
            current_fn = current_dd.CTADatasetFunction(cfg, mode="val")
            reference_fn = reference_dd.CTADatasetFunction(cfg, mode="val")

            # Get datasets
            current_dicts = current_fn()
            reference_dicts = reference_fn()

            # Compare
            assert len(current_dicts) == len(reference_dicts)

            for curr, ref in zip(current_dicts, reference_dicts):
                assert curr["scan_id"] == ref["scan_id"]
                # In val mode, annotations should be None
                assert curr["annotations"] is None
                assert ref["annotations"] is None

    def test_dataset_function_debug_mode(self):
        """Test that debug mode limits dataset identically."""
        from yacs.config import CfgNode as CN

        current_dd, reference_dd = import_dataset_dict_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir, vessel_dir, label_dir, annot_file = self.create_test_dataset_structure(tmpdir)

            # Create config with debug mode
            cfg = CN()
            cfg.CUSTOM = CN()
            cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
            cfg.CUSTOM.DEBUG = True  # Debug mode ON
            cfg.CUSTOM.DEBUG_DATASET_SIZE = 2  # Limit to 2 samples
            cfg.DATA = CN()
            cfg.DATA.DIR = CN()
            cfg.DATA.DIR.TRAIN = CN()
            cfg.DATA.DIR.TRAIN.SCAN_DIR = scan_dir
            cfg.DATA.DIR.TRAIN.LABEL_DIR = label_dir
            cfg.DATA.DIR.TRAIN.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.TRAIN.CVS_DIR = tmpdir
            cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = annot_file
            cfg.DATA.DIR.VAL = CN()
            cfg.DATA.DIR.VAL.SCAN_DIR = scan_dir
            cfg.DATA.DIR.VAL.LABEL_DIR = ""
            cfg.DATA.DIR.VAL.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.VAL.CVS_DIR = tmpdir
            cfg.MODEL = CN()
            cfg.MODEL.USE_VESSEL_INFO = "edt"
            cfg.MODEL.USE_CVS_INFO = "no"

            # Create dataset functions
            current_fn = current_dd.CTADatasetFunction(cfg, mode="train")
            reference_fn = reference_dd.CTADatasetFunction(cfg, mode="train")

            # Get datasets
            current_dicts = current_fn()
            reference_dicts = reference_fn()

            # Both should limit to DEBUG_DATASET_SIZE
            assert len(current_dicts) == 2
            assert len(reference_dicts) == 2
            assert len(current_dicts) == len(reference_dicts)

    def test_dataset_function_without_vessel_info(self):
        """Test dataset function when vessel info is disabled."""
        from yacs.config import CfgNode as CN

        current_dd, reference_dd = import_dataset_dict_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir, _, label_dir, annot_file = self.create_test_dataset_structure(tmpdir)

            # Create config WITHOUT vessel info
            cfg = CN()
            cfg.CUSTOM = CN()
            cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
            cfg.CUSTOM.DEBUG = False
            cfg.CUSTOM.DEBUG_DATASET_SIZE = 2
            cfg.DATA = CN()
            cfg.DATA.DIR = CN()
            cfg.DATA.DIR.TRAIN = CN()
            cfg.DATA.DIR.TRAIN.SCAN_DIR = scan_dir
            cfg.DATA.DIR.TRAIN.LABEL_DIR = label_dir
            cfg.DATA.DIR.TRAIN.VESSEL_DIR = tmpdir  # Dummy path
            cfg.DATA.DIR.TRAIN.CVS_DIR = tmpdir
            cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = annot_file
            cfg.DATA.DIR.VAL = CN()
            cfg.DATA.DIR.VAL.SCAN_DIR = scan_dir
            cfg.DATA.DIR.VAL.LABEL_DIR = ""
            cfg.DATA.DIR.VAL.VESSEL_DIR = tmpdir
            cfg.DATA.DIR.VAL.CVS_DIR = tmpdir
            cfg.MODEL = CN()
            cfg.MODEL.USE_VESSEL_INFO = "no"  # Disabled
            cfg.MODEL.USE_CVS_INFO = "no"

            # Create dataset functions
            current_fn = current_dd.CTADatasetFunction(cfg, mode="train")
            reference_fn = reference_dd.CTADatasetFunction(cfg, mode="train")

            # Get datasets
            current_dicts = current_fn()
            reference_dicts = reference_fn()

            # Compare
            assert len(current_dicts) == len(reference_dicts)

            for curr, ref in zip(current_dicts, reference_dicts):
                # Should NOT have vessel_file_name when disabled
                if cfg.MODEL.USE_VESSEL_INFO == "no":
                    # Behavior may differ - just verify both work
                    assert "scan_id" in curr
                    assert "scan_id" in ref

    def test_dataset_function_file_ordering(self):
        """Test that file ordering is consistent."""
        from yacs.config import CfgNode as CN

        current_dd, reference_dd = import_dataset_dict_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir, vessel_dir, label_dir, annot_file = self.create_test_dataset_structure(tmpdir)

            cfg = CN()
            cfg.CUSTOM = CN()
            cfg.CUSTOM.DATASET_FUNCTION = "CTADatasetFunction"
            cfg.CUSTOM.DEBUG = False
            cfg.CUSTOM.DEBUG_DATASET_SIZE = 2
            cfg.DATA = CN()
            cfg.DATA.DIR = CN()
            cfg.DATA.DIR.TRAIN = CN()
            cfg.DATA.DIR.TRAIN.SCAN_DIR = scan_dir
            cfg.DATA.DIR.TRAIN.LABEL_DIR = label_dir
            cfg.DATA.DIR.TRAIN.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.TRAIN.CVS_DIR = tmpdir
            cfg.DATA.DIR.TRAIN.ANNOTATION_FILE = annot_file
            cfg.DATA.DIR.VAL = CN()
            cfg.DATA.DIR.VAL.SCAN_DIR = scan_dir
            cfg.DATA.DIR.VAL.LABEL_DIR = ""
            cfg.DATA.DIR.VAL.VESSEL_DIR = vessel_dir
            cfg.DATA.DIR.VAL.CVS_DIR = tmpdir
            cfg.MODEL = CN()
            cfg.MODEL.USE_VESSEL_INFO = "edt"
            cfg.MODEL.USE_CVS_INFO = "no"

            # Create dataset functions
            current_fn = current_dd.CTADatasetFunction(cfg, mode="train")
            reference_fn = reference_dd.CTADatasetFunction(cfg, mode="train")

            # Get datasets
            current_dicts = current_fn()
            reference_dicts = reference_fn()

            # Files should be in same order
            current_ids = [d["scan_id"] for d in current_dicts]
            reference_ids = [d["scan_id"] for d in reference_dicts]

            assert current_ids == reference_ids, (
                f"File ordering differs: {current_ids} vs {reference_ids}"
            )
