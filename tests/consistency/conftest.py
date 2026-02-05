"""
Pytest configuration and shared fixtures for consistency tests.

This module provides common fixtures and utilities for comparing outputs
between the refactored code and the reference implementation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest


# Paths to both project versions
CURRENT_PROJECT_ROOT = Path("/projects/vig/alberto/medical/exploration/deform")
REFERENCE_PROJECT_ROOT = Path("/projects/vig/alberto/medical/deform-aneurysm-detection")


@pytest.fixture
def current_project_path():
    """Path to the current (refactored) project."""
    return CURRENT_PROJECT_ROOT


@pytest.fixture
def reference_project_path():
    """Path to the reference (original) project."""
    return REFERENCE_PROJECT_ROOT


@pytest.fixture
def add_current_to_path():
    """Add current project to Python path."""
    path = str(CURRENT_PROJECT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    yield
    if path in sys.path:
        sys.path.remove(path)


@pytest.fixture
def add_reference_to_path():
    """Add reference project to Python path."""
    path = str(REFERENCE_PROJECT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    yield
    if path in sys.path:
        sys.path.remove(path)


@pytest.fixture
def mock_3d_image():
    """Generate a mock 3D medical image for testing."""
    np.random.seed(42)
    # Typical CTA scan dimensions (smaller for testing)
    shape = (64, 128, 128)  # D, H, W
    image = np.random.randn(*shape).astype(np.float32) * 100 + 50
    return image


@pytest.fixture
def mock_vessel_mask():
    """Generate a mock vessel segmentation mask."""
    np.random.seed(42)
    shape = (64, 128, 128)  # D, H, W
    mask = (np.random.rand(*shape) > 0.9).astype(np.float32)
    return mask


@pytest.fixture
def mock_annotations():
    """Generate mock aneurysm annotations."""
    np.random.seed(42)
    # Format: [coordX, coordY, coordZ, w, h, d, lesion_type]
    annotations = np.array([
        [64.0, 64.0, 32.0, 5.0, 5.0, 5.0, "aneurysm"],
        [80.0, 80.0, 40.0, 3.0, 3.0, 3.0, "aneurysm"],
        [50.0, 50.0, 20.0, 4.0, 4.0, 4.0, "non_aneurysm"],
    ])
    return annotations


@pytest.fixture
def mock_dataset_dict(tmp_path):
    """Generate a mock dataset dictionary."""
    return {
        "scan_id": "test_scan_001.nii.gz",
        "file_name": str(tmp_path / "scan.nii.gz"),
        "vessel_file_name": str(tmp_path / "vessel.nii.gz"),
        "cvs_file_name": str(tmp_path / "cvs.nii.gz"),
        "annotations": np.array([
            [64.0, 64.0, 32.0, 5.0, 5.0, 5.0, "aneurysm"],
            [80.0, 80.0, 40.0, 3.0, 3.0, 3.0, "aneurysm"],
        ]),
    }


def assert_arrays_close(arr1, arr2, rtol=1e-5, atol=1e-8, name="array"):
    """
    Assert that two arrays are close in value.

    Args:
        arr1: First array
        arr2: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
    """
    assert arr1.shape == arr2.shape, (
        f"{name} shape mismatch: {arr1.shape} vs {arr2.shape}"
    )
    assert arr1.dtype == arr2.dtype, (
        f"{name} dtype mismatch: {arr1.dtype} vs {arr2.dtype}"
    )
    np.testing.assert_allclose(
        arr1, arr2, rtol=rtol, atol=atol,
        err_msg=f"{name} values differ"
    )


def assert_dicts_close(dict1, dict2, rtol=1e-5, atol=1e-8):
    """
    Assert that two dictionaries containing arrays are close.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        rtol: Relative tolerance for array comparison
        atol: Absolute tolerance for array comparison
    """
    assert set(dict1.keys()) == set(dict2.keys()), (
        f"Dictionary keys differ: {dict1.keys()} vs {dict2.keys()}"
    )

    for key in dict1.keys():
        val1, val2 = dict1[key], dict2[key]

        if isinstance(val1, np.ndarray):
            assert_arrays_close(val1, val2, rtol=rtol, atol=atol, name=key)
        elif isinstance(val1, (list, tuple)):
            assert len(val1) == len(val2), f"Length mismatch for {key}"
            for i, (v1, v2) in enumerate(zip(val1, val2)):
                if isinstance(v1, np.ndarray):
                    assert_arrays_close(v1, v2, rtol=rtol, atol=atol,
                                       name=f"{key}[{i}]")
        else:
            assert val1 == val2, f"Value mismatch for {key}: {val1} vs {val2}"


@pytest.fixture
def numpy_seed():
    """Set numpy random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)
