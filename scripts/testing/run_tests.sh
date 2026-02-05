#!/bin/bash
# Script to run consistency tests for deformable aneurysm detection project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Deformable Aneurysm Detection Tests"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -d "tests" ]; then
    echo -e "${RED}Error: tests directory not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}Warning: pytest is not installed${NC}"
    echo "Installing test dependencies..."
    pip install -r tests/requirements_test.txt
fi

# Parse command line arguments
TEST_PATH="tests/"
VERBOSE=""
COVERAGE=""
PARALLEL=""
STOP_ON_FAIL=""
KEYWORD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=src --cov-report=html --cov-report=term"
            shift
            ;;
        -p|--parallel)
            PARALLEL="-n auto"
            shift
            ;;
        -x|--stop-on-fail)
            STOP_ON_FAIL="-x"
            shift
            ;;
        -k|--keyword)
            KEYWORD="-k $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose        Show verbose output"
            echo "  -c, --coverage       Generate coverage report"
            echo "  -p, --parallel       Run tests in parallel"
            echo "  -x, --stop-on-fail   Stop at first failure"
            echo "  -k, --keyword EXPR   Only run tests matching keyword"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all tests"
            echo "  $0 -v -c                   # Run with verbose output and coverage"
            echo "  $0 -k consistency          # Run only consistency tests"
            echo "  $0 -x -v                   # Stop at first failure with verbose output"
            exit 0
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Run the tests
echo -e "${GREEN}Running tests...${NC}"
echo "Test path: $TEST_PATH"
echo ""

pytest $TEST_PATH $VERBOSE $COVERAGE $PARALLEL $STOP_ON_FAIL $KEYWORD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"

    if [ -n "$COVERAGE" ]; then
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
