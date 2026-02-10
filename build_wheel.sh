#!/bin/bash
# Build wheel for TheMol
# Usage: bash build_wheel.sh

set -e

echo "=========================================="
echo "Building TheMol wheel"
echo "=========================================="

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Get Python and PyTorch info
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "unknown")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cpu")

echo "Python version: ${PYTHON_VERSION}"
echo "PyTorch version: ${TORCH_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"

# Build wheel
pip install build wheel setuptools --upgrade
python -m build --wheel

# Rename wheel with version info
WHEEL_FILE=$(ls dist/*.whl | head -1)
if [ -n "$WHEEL_FILE" ]; then
    NEW_NAME=$(echo "$WHEEL_FILE" | sed "s/themol-/themol-/" | sed "s/-py3/-cu${CUDA_VERSION}torch${TORCH_VERSION}-py${PYTHON_VERSION//.}/")
    # Keep original name for now (standard naming)
    echo ""
    echo "=========================================="
    echo "Build successful!"
    echo "Wheel file: ${WHEEL_FILE}"
    echo "=========================================="
    echo ""
    echo "To install:"
    echo "  pip install ${WHEEL_FILE}"
    echo ""
    echo "To upload to GitHub Releases:"
    echo "  gh release upload v0.1.0 ${WHEEL_FILE}"
fi
