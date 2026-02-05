"""
End-to-end consistency tests for dataloader module.

Tests that compare actual data loading behavior between current and reference implementations.
"""

import sys
import importlib
import numpy as np

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
)


def import_dataloader_modules():
    """Import dataloader from both versions."""
    # Current version
    sys.path.insert(0, str(CURRENT_PROJECT_ROOT))
    modules_to_clear = ['src.dataset.dataloader', 'src.dataset', 'src']
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    current_dataloader = importlib.import_module('src.dataset.dataloader')
    sys.path.remove(str(CURRENT_PROJECT_ROOT))

    # Reference version
    sys.path.insert(0, str(REFERENCE_PROJECT_ROOT))
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    reference_dataloader = importlib.import_module('src.dataset.dataloader')
    sys.path.remove(str(REFERENCE_PROJECT_ROOT))

    return current_dataloader, reference_dataloader


class TestDataLoaderEndToEnd:
    """E2E tests for dataloader module comparing actual behavior."""

    def test_get_dataset_dicts_with_mock_catalog(self):
        """Test that get_dataset_dicts produces identical output for same input."""
        current_dl, reference_dl = import_dataloader_modules()

        # Create mock dataset
        mock_dicts = [
            {"scan_id": "scan_001", "file_name": "/tmp/scan_001.nii.gz"},
            {"scan_id": "scan_002", "file_name": "/tmp/scan_002.nii.gz"},
            {"scan_id": "scan_003", "file_name": "/tmp/scan_003.nii.gz"},
        ]

        try:
            from detectron2.data import DatasetCatalog

            # Register mock dataset
            if "test_e2e_dataloader" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_dataloader")

            DatasetCatalog.register("test_e2e_dataloader", lambda: mock_dicts.copy())

            # Get from both versions
            current_result = current_dl.get_dataset_dicts("test_e2e_dataloader")
            reference_result = reference_dl.get_dataset_dicts("test_e2e_dataloader")

            # Compare
            assert len(current_result) == len(reference_result), (
                f"Different lengths: {len(current_result)} vs {len(reference_result)}"
            )

            for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
                assert curr == ref, f"Mismatch at index {i}: {curr} vs {ref}"

        finally:
            if "test_e2e_dataloader" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_dataloader")

    def test_get_dataset_dicts_multiple_datasets(self):
        """Test that combining multiple datasets works identically."""
        current_dl, reference_dl = import_dataloader_modules()

        mock_dicts_1 = [
            {"scan_id": "scan_001", "file_name": "/tmp/scan_001.nii.gz"},
        ]
        mock_dicts_2 = [
            {"scan_id": "scan_002", "file_name": "/tmp/scan_002.nii.gz"},
        ]

        try:
            from detectron2.data import DatasetCatalog

            # Register mock datasets
            for name in ["test_e2e_ds1", "test_e2e_ds2"]:
                if name in DatasetCatalog.list():
                    DatasetCatalog.remove(name)

            DatasetCatalog.register("test_e2e_ds1", lambda: mock_dicts_1.copy())
            DatasetCatalog.register("test_e2e_ds2", lambda: mock_dicts_2.copy())

            # Get combined datasets
            current_result = current_dl.get_dataset_dicts(["test_e2e_ds1", "test_e2e_ds2"])
            reference_result = reference_dl.get_dataset_dicts(["test_e2e_ds1", "test_e2e_ds2"])

            # Compare
            assert len(current_result) == len(reference_result) == 2
            for curr, ref in zip(current_result, reference_result):
                assert curr == ref

        finally:
            for name in ["test_e2e_ds1", "test_e2e_ds2"]:
                if name in DatasetCatalog.list():
                    DatasetCatalog.remove(name)

    def test_train_loader_from_config_structure(self):
        """Test that train loader config produces identical structure."""
        from yacs.config import CfgNode as CN

        current_dl, reference_dl = import_dataloader_modules()

        # Create config
        cfg = CN()
        cfg.DATASETS = CN()
        cfg.DATASETS.TRAIN = ["mock_dataset"]
        cfg.DATALOADER = CN()
        cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER = CN()
        cfg.SOLVER.SCANS_PER_BATCH = 2
        cfg.CUSTOM = CN()
        cfg.CUSTOM.DEBUG = False

        # Mock dataset and mapper
        mock_dataset = [{"scan_id": f"scan_{i:03d}"} for i in range(10)]
        mock_mapper = lambda x: x

        # Get configs from both
        current_config = current_dl._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )
        reference_config = reference_dl._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )

        # Compare structure
        assert set(current_config.keys()) == set(reference_config.keys()), (
            f"Config keys differ: {current_config.keys()} vs {reference_config.keys()}"
        )

        # Compare specific values
        assert current_config["total_batch_size"] == reference_config["total_batch_size"]
        assert current_config["num_workers"] == reference_config["num_workers"]
        assert current_config["aspect_ratio_grouping"] == reference_config["aspect_ratio_grouping"]
        assert len(current_config["dataset"]) == len(reference_config["dataset"])

    def test_test_loader_from_config_structure(self):
        """Test that test loader config is identical."""
        from yacs.config import CfgNode as CN
        from detectron2.data import DatasetCatalog

        current_dl, reference_dl = import_dataloader_modules()

        cfg = CN()
        cfg.DATALOADER = CN()
        cfg.DATALOADER.NUM_WORKERS = 2

        # Register mock dataset
        mock_dataset = [{"scan_id": f"scan_{i:03d}"} for i in range(5)]

        try:
            if "test_e2e_loader" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_loader")

            DatasetCatalog.register("test_e2e_loader", lambda: mock_dataset)

            mock_mapper = lambda x: x

            # Get configs
            current_config = current_dl._test_loader_from_config(
                cfg, "test_e2e_loader", mock_mapper
            )
            reference_config = reference_dl._test_loader_from_config(
                cfg, "test_e2e_loader", mock_mapper
            )

            # Compare
            assert set(current_config.keys()) == set(reference_config.keys())
            assert current_config["num_workers"] == reference_config["num_workers"]
            assert len(current_config["dataset"]) == len(reference_config["dataset"])

        finally:
            if "test_e2e_loader" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_loader")

    def test_train_loader_with_debug_mode(self):
        """Test that debug mode is handled identically."""
        from yacs.config import CfgNode as CN

        current_dl, reference_dl = import_dataloader_modules()

        # Create config with debug mode
        cfg = CN()
        cfg.DATASETS = CN()
        cfg.DATASETS.TRAIN = ["mock_dataset"]
        cfg.DATALOADER = CN()
        cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.SOLVER = CN()
        cfg.SOLVER.SCANS_PER_BATCH = 1
        cfg.CUSTOM = CN()
        cfg.CUSTOM.DEBUG = True  # Debug mode ON

        mock_dataset = [{"scan_id": f"scan_{i:03d}"} for i in range(10)]
        mock_mapper = lambda x: x

        # Get configs
        current_config = current_dl._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )
        reference_config = reference_dl._train_loader_from_config(
            cfg, mock_mapper, dataset=mock_dataset
        )

        # Verify sampler behavior is same
        # In debug mode, shuffle should be False
        assert type(current_config["sampler"]) == type(reference_config["sampler"])

    def test_empty_dataset_error_handling(self):
        """Test that both versions handle empty datasets identically."""
        current_dl, reference_dl = import_dataloader_modules()

        try:
            from detectron2.data import DatasetCatalog

            if "empty_e2e_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("empty_e2e_dataset")

            DatasetCatalog.register("empty_e2e_dataset", lambda: [])

            # Both should raise AssertionError
            current_raised = False
            reference_raised = False

            try:
                current_dl.get_dataset_dicts("empty_e2e_dataset")
            except AssertionError:
                current_raised = True

            try:
                reference_dl.get_dataset_dicts("empty_e2e_dataset")
            except AssertionError:
                reference_raised = True

            assert current_raised, "Current version did not raise error for empty dataset"
            assert reference_raised, "Reference version did not raise error for empty dataset"

        finally:
            if "empty_e2e_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove("empty_e2e_dataset")

    def test_dataset_order_preservation(self):
        """Test that dataset order is preserved identically."""
        current_dl, reference_dl = import_dataloader_modules()

        # Create ordered mock dataset
        mock_dicts = [
            {"scan_id": f"scan_{i:03d}", "order": i}
            for i in range(20)
        ]

        try:
            from detectron2.data import DatasetCatalog

            if "test_e2e_order" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_order")

            DatasetCatalog.register("test_e2e_order", lambda: mock_dicts.copy())

            # Get from both
            current_result = current_dl.get_dataset_dicts("test_e2e_order")
            reference_result = reference_dl.get_dataset_dicts("test_e2e_order")

            # Verify order is preserved
            for i, (curr, ref) in enumerate(zip(current_result, reference_result)):
                assert curr["order"] == i
                assert ref["order"] == i
                assert curr["order"] == ref["order"]

        finally:
            if "test_e2e_order" in DatasetCatalog.list():
                DatasetCatalog.remove("test_e2e_order")
