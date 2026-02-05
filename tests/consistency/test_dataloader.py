"""
Consistency tests for dataloader module.

Tests that verify data loading utilities produce identical outputs
in both versions.
"""

import sys
import importlib
import numpy as np
import pytest

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
)


def import_dataloader_module(project_root):
    """
    Import dataloader module from specified project.

    Args:
        project_root: Path to project root

    Returns:
        Imported module
    """
    sys.path.insert(0, str(project_root))
    try:
        # Clear cached modules
        modules_to_clear = [
            'src.dataset.dataloader',
            'src.dataset',
            'src',
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        module = importlib.import_module('src.dataset.dataloader')
        return module
    finally:
        sys.path.remove(str(project_root))


class TestGetDatasetDictsConsistency:
    """Test get_dataset_dicts function consistency."""

    def test_single_dataset_name(self):
        """Test get_dataset_dicts with a single dataset name."""
        current_mod = import_dataloader_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataloader_module(REFERENCE_PROJECT_ROOT)

        # Mock dataset dicts
        mock_dicts = [
            {"scan_id": "scan_001", "file_name": "/tmp/scan_001.nii.gz"},
            {"scan_id": "scan_002", "file_name": "/tmp/scan_002.nii.gz"},
        ]

        # Register mock dataset in both versions
        try:
            from detectron2.data import DatasetCatalog

            # Clear any existing registrations
            if "test_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("test_dataset")

            DatasetCatalog.register("test_dataset", lambda: mock_dicts.copy())

            # Get dataset dicts from both versions
            current_result = current_mod.get_dataset_dicts("test_dataset")
            reference_result = reference_mod.get_dataset_dicts("test_dataset")

            # Compare
            assert len(current_result) == len(reference_result)
            for curr, ref in zip(current_result, reference_result):
                assert curr == ref

        finally:
            # Cleanup
            if "test_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("test_dataset")

    def test_multiple_dataset_names(self):
        """Test get_dataset_dicts with multiple dataset names."""
        current_mod = import_dataloader_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataloader_module(REFERENCE_PROJECT_ROOT)

        # Mock dataset dicts
        mock_dicts_1 = [
            {"scan_id": "scan_001", "file_name": "/tmp/scan_001.nii.gz"},
        ]
        mock_dicts_2 = [
            {"scan_id": "scan_002", "file_name": "/tmp/scan_002.nii.gz"},
        ]

        try:
            from detectron2.data import DatasetCatalog

            # Register mock datasets
            for name in ["test_dataset_1", "test_dataset_2"]:
                if name in DatasetCatalog.list():
                    DatasetCatalog.remove(name)

            DatasetCatalog.register("test_dataset_1", lambda: mock_dicts_1.copy())
            DatasetCatalog.register("test_dataset_2", lambda: mock_dicts_2.copy())

            # Get combined dataset dicts
            current_result = current_mod.get_dataset_dicts(
                ["test_dataset_1", "test_dataset_2"]
            )
            reference_result = reference_mod.get_dataset_dicts(
                ["test_dataset_1", "test_dataset_2"]
            )

            # Compare
            assert len(current_result) == len(reference_result) == 2
            for curr, ref in zip(current_result, reference_result):
                assert curr == ref

        finally:
            # Cleanup
            for name in ["test_dataset_1", "test_dataset_2"]:
                if name in DatasetCatalog.list():
                    DatasetCatalog.remove(name)

    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises assertion error in both versions."""
        current_mod = import_dataloader_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataloader_module(REFERENCE_PROJECT_ROOT)

        try:
            from detectron2.data import DatasetCatalog

            # Register empty dataset
            if "empty_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("empty_dataset")

            DatasetCatalog.register("empty_dataset", lambda: [])

            # Both should raise AssertionError
            with pytest.raises(AssertionError):
                current_mod.get_dataset_dicts("empty_dataset")

            with pytest.raises(AssertionError):
                reference_mod.get_dataset_dicts("empty_dataset")

        finally:
            if "empty_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("empty_dataset")


class TestTrainLoaderConfig:
    """Test training loader configuration consistency."""

    def test_train_loader_config_structure(self):
        """Test that _train_loader_from_config produces consistent structure."""
        current_mod = import_dataloader_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataloader_module(REFERENCE_PROJECT_ROOT)

        # Create minimal mock config
        from yacs.config import CfgNode as CN

        cfg = CN()
        cfg.DATASETS = CN()
        cfg.DATASETS.TRAIN = ["mock_train"]

        cfg.DATALOADER = CN()
        cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
        cfg.DATALOADER.NUM_WORKERS = 4

        cfg.SOLVER = CN()
        cfg.SOLVER.SCANS_PER_BATCH = 2

        cfg.CUSTOM = CN()
        cfg.CUSTOM.DEBUG = False

        # Mock dataset
        mock_dataset = [
            {"scan_id": f"scan_{i:03d}"} for i in range(10)
        ]

        # Mock mapper
        mock_mapper = lambda x: x

        # Get config from both versions
        current_config = current_mod._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )
        reference_config = reference_mod._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )

        # Compare configuration structure
        assert set(current_config.keys()) == set(reference_config.keys())
        assert current_config["total_batch_size"] == reference_config["total_batch_size"]
        assert current_config["num_workers"] == reference_config["num_workers"]
        assert (
            current_config["aspect_ratio_grouping"]
            == reference_config["aspect_ratio_grouping"]
        )
        assert len(current_config["dataset"]) == len(reference_config["dataset"])


class TestTestLoaderConfig:
    """Test test loader configuration consistency."""

    def test_test_loader_config_structure(self):
        """Test that _test_loader_from_config produces consistent structure."""
        current_mod = import_dataloader_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_dataloader_module(REFERENCE_PROJECT_ROOT)

        # Create minimal mock config
        from yacs.config import CfgNode as CN

        cfg = CN()
        cfg.DATALOADER = CN()
        cfg.DATALOADER.NUM_WORKERS = 4

        # Mock dataset
        try:
            from detectron2.data import DatasetCatalog

            if "test_val" in DatasetCatalog.list():
                DatasetCatalog.remove("test_val")

            mock_dataset = [
                {"scan_id": f"scan_{i:03d}"} for i in range(5)
            ]
            DatasetCatalog.register("test_val", lambda: mock_dataset)

            # Mock mapper
            mock_mapper = lambda x: x

            # Get config from both versions
            current_config = current_mod._test_loader_from_config(
                cfg, "test_val", mock_mapper
            )
            reference_config = reference_mod._test_loader_from_config(
                cfg, "test_val", mock_mapper
            )

            # Compare configuration structure
            assert set(current_config.keys()) == set(reference_config.keys())
            assert current_config["num_workers"] == reference_config["num_workers"]
            assert len(current_config["dataset"]) == len(reference_config["dataset"])

        finally:
            if "test_val" in DatasetCatalog.list():
                DatasetCatalog.remove("test_val")
