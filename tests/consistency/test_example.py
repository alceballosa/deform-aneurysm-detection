"""
Example consistency test demonstrating the test pattern.

This is a simple example showing how consistency tests work.
"""

import sys
import importlib
import numpy as np
import pytest

from .conftest import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_PROJECT_ROOT,
)


def import_config_module(project_root):
    """Import config module from specified project."""
    sys.path.insert(0, str(project_root))
    try:
        # Clear cached modules
        if 'src.config' in sys.modules:
            del sys.modules['src.config']
        if 'src' in sys.modules:
            del sys.modules['src']

        module = importlib.import_module('src.config')
        return module
    finally:
        sys.path.remove(str(project_root))


class TestExampleConsistency:
    """
    Example test class demonstrating consistency testing pattern.

    This class shows the basic structure of a consistency test:
    1. Import the same module from both projects
    2. Call the same function/class with identical inputs
    3. Compare outputs to ensure they're identical
    """

    def test_module_import(self):
        """Test that we can import modules from both projects."""
        # This test verifies the test infrastructure itself
        current_mod = import_config_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_config_module(REFERENCE_PROJECT_ROOT)

        # Both modules should be imported successfully
        assert current_mod is not None
        assert reference_mod is not None

    def test_basic_constant(self):
        """Test that a constant value is the same in both versions."""
        current_mod = import_config_module(CURRENT_PROJECT_ROOT)
        reference_mod = import_config_module(REFERENCE_PROJECT_ROOT)

        # Example: compare a constant (if it exists in your config module)
        # This demonstrates comparing simple values
        # Note: Adjust this based on actual constants in your config module

        # For demonstration, we just check that both modules have similar structure
        current_attrs = set(dir(current_mod))
        reference_attrs = set(dir(reference_mod))

        # Check that the modules have similar attributes
        # (This is a weak test but demonstrates the concept)
        assert len(current_attrs & reference_attrs) > 0, (
            "Modules should have some common attributes"
        )

    def test_array_comparison_example(self):
        """Example of comparing numpy arrays between versions."""
        from .conftest import assert_arrays_close

        # Simulate two outputs from different versions
        # In real tests, these would come from calling functions
        current_output = np.array([1.0, 2.0, 3.0])
        reference_output = np.array([1.0, 2.0, 3.0])

        # This should pass
        assert_arrays_close(current_output, reference_output, name="example_array")

    def test_array_comparison_with_tolerance(self):
        """Example of comparing arrays with floating point tolerance."""
        from .conftest import assert_arrays_close

        # Simulate small numerical differences due to floating point
        current_output = np.array([1.0000001, 2.0, 3.0])
        reference_output = np.array([1.0, 2.0, 3.0])

        # This should pass with appropriate tolerance
        assert_arrays_close(
            current_output,
            reference_output,
            rtol=1e-5,
            atol=1e-8,
            name="example_array_with_tolerance"
        )

    def test_dictionary_comparison_example(self):
        """Example of comparing dictionaries containing arrays."""
        from .conftest import assert_dicts_close

        # Simulate outputs from different versions
        current_output = {
            "image": np.random.randn(10, 10).astype(np.float32),
            "label": np.array([0, 1, 1, 0]),
            "metadata": "scan_001",
        }

        # Copy to simulate reference output
        reference_output = {
            "image": current_output["image"].copy(),
            "label": current_output["label"].copy(),
            "metadata": current_output["metadata"],
        }

        # This should pass
        assert_dicts_close(current_output, reference_output)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_deterministic_with_seed(self, seed):
        """
        Example of testing deterministic behavior with different seeds.

        This pattern is useful for testing random operations that should
        be reproducible with a fixed seed.
        """
        # Set seed and generate random array (current version)
        np.random.seed(seed)
        current_output = np.random.randn(5, 5)

        # Set same seed and generate again (reference version)
        np.random.seed(seed)
        reference_output = np.random.randn(5, 5)

        # Should be identical with same seed
        from .conftest import assert_arrays_close
        assert_arrays_close(current_output, reference_output, name="random_output")


class TestExampleErrorHandling:
    """Example of testing error handling consistency."""

    def test_error_raised_consistently(self):
        """Test that both versions raise the same error."""
        # Example: both versions should raise ValueError for invalid input

        def invalid_operation_current():
            raise ValueError("Invalid input")

        def invalid_operation_reference():
            raise ValueError("Invalid input")

        # Both should raise ValueError
        with pytest.raises(ValueError):
            invalid_operation_current()

        with pytest.raises(ValueError):
            invalid_operation_reference()

    def test_error_messages_consistent(self):
        """Test that error messages are consistent."""

        def operation_current(value):
            if value < 0:
                raise ValueError("Value must be non-negative")
            return value * 2

        def operation_reference(value):
            if value < 0:
                raise ValueError("Value must be non-negative")
            return value * 2

        # Valid input - should work
        assert operation_current(5) == operation_reference(5)

        # Invalid input - should raise same error
        with pytest.raises(ValueError, match="non-negative"):
            operation_current(-1)

        with pytest.raises(ValueError, match="non-negative"):
            operation_reference(-1)
