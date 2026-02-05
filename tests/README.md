# Test Suite for Deformable Aneurysm Detection

This directory contains tests to ensure code quality and consistency during refactoring.

## Test Structure

```
tests/
├── __init__.py
├── README.md
├── requirements_test.txt
└── consistency/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures and utilities
    ├── test_dataloader.py    # Tests for dataloader module
    ├── test_dataset_dict.py  # Tests for dataset_dict module
    ├── test_dataset_mapper.py # Tests for dataset_mapper module
    └── test_crop2.py         # Tests for crop2 module
```

## Consistency Tests

The `consistency/` directory contains tests that compare outputs between:
- **Current version**: `/projects/vig/alberto/medical/exploration/deform` (refactored code)
- **Reference version**: `/projects/vig/alberto/medical/deform-aneurysm-detection` (original implementation)

These tests ensure that refactoring doesn't change the behavior of the code.

### What is Tested

#### Dataset Module (`src/dataset/`)
- **dataloader.py**: Data loading utilities, batch construction, dataset histograms
- **dataset_dict.py**: Dataset registration, path resolution, CTADatasetFunction
- **dataset_mapper.py**: Image loading, normalization, augmentation configuration
- **crop2.py**: Patch extraction, spatial augmentations, cropping behavior

### Test Categories

1. **Initialization Tests**: Verify that classes are initialized with the same parameters
2. **Output Structure Tests**: Ensure outputs have identical structure and keys
3. **Numerical Consistency Tests**: Compare numerical outputs using numpy array comparisons
4. **Edge Case Tests**: Test boundary conditions and error handling

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements_test.txt
```

### Run All Tests

```bash
# From the project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Suites

```bash
# Run only consistency tests
pytest tests/consistency/

# Run tests for a specific module
pytest tests/consistency/test_dataset_mapper.py

# Run a specific test class
pytest tests/consistency/test_dataset_mapper.py::TestNormalizationConsistency

# Run a specific test
pytest tests/consistency/test_dataset_mapper.py::TestNormalizationConsistency::test_normalize_window_based
```

### Useful Pytest Options

```bash
# Stop at first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Only run tests matching a keyword
pytest tests/ -k "consistency"

# Show slowest tests
pytest tests/ --durations=10
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Consistency Test Pattern

```python
def test_function_consistency():
    """Test that function produces identical outputs."""
    # Import both versions
    current_mod = import_module(CURRENT_PROJECT_ROOT)
    reference_mod = import_module(REFERENCE_PROJECT_ROOT)

    # Create test inputs
    test_input = ...

    # Get outputs from both versions
    current_output = current_mod.function(test_input)
    reference_output = reference_mod.function(test_input)

    # Compare outputs
    assert_arrays_close(current_output, reference_output)
```

### Helper Functions

Available in `conftest.py`:
- `assert_arrays_close(arr1, arr2, ...)`: Compare numpy arrays with tolerance
- `assert_dicts_close(dict1, dict2, ...)`: Compare dictionaries containing arrays
- Fixtures for mock data: `mock_3d_image`, `mock_vessel_mask`, `mock_annotations`

## Continuous Integration

These tests should be run:
1. Before committing changes
2. In CI/CD pipelines
3. Before merging pull requests

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. Both project directories exist and are accessible
2. Required dependencies are installed
3. Python path is correctly set up

### Numerical Differences

Small numerical differences may occur due to:
- Floating point precision
- Random seed differences
- Library version differences

Adjust tolerances in `assert_arrays_close()` if needed:
```python
assert_arrays_close(arr1, arr2, rtol=1e-5, atol=1e-8)
```

### Module Caching

If changes aren't reflected, clear Python's module cache:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Future Test Additions

Consider adding tests for:
- [ ] Model architectures (`src/models/`)
- [ ] Preprocessing pipelines (`src/preprocess/`)
- [ ] Post-processing (`src/postprocess/`)
- [ ] Transforms (`src/transform/`)
- [ ] Training logic (`src/train_net.py`)
- [ ] FROC evaluation (`src/froc_overlap_2.py`)
- [ ] Integration tests with real data
- [ ] Performance benchmarks

## Contact

For questions about the test suite, contact the development team.
