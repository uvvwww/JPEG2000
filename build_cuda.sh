#!/bin/bash
# Quick start script for building with CUDA support

echo "========================================="
echo "CUDA DWT Acceleration Quick Start"
echo "========================================="
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found!"
    echo "Please install CUDA Toolkit first:"
    echo "  - Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "  - Or download from: https://developer.nvidia.com/cuda-downloads"
    echo ""
    exit 1
fi

echo "✓ CUDA compiler found: $(nvcc --version | head -n 1)"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. No NVIDIA GPU detected?"
    echo "CUDA code will compile but may not run."
    echo ""
else
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
    echo ""
fi

# Build with CUDA
echo "Building with CUDA support..."
echo "Command: make clean && make ENABLE_CUDA=1 -j"
echo ""

make clean
if make ENABLE_CUDA=1 -j; then
    echo ""
    echo "========================================="
    echo "✓ Build successful!"
    echo "========================================="
    echo ""
    echo "Try it out:"
    echo "  ./build/j2k_encode_pnm input/lena.ppm output/lena_cuda.j2k"
    echo "  ./build/j2k_decode_pnm output/lena_cuda.j2k output/lena_cuda_decoded.ppm"
    echo ""
    echo "Profile it:"
    echo "  ./build/j2k_encode_profile input/03.ppm output/03_cuda.j2k"
    echo ""
    echo "Look for 'CUDA DWT initialized' in output to confirm GPU usage."
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Build failed!"
    echo "========================================="
    echo ""
    echo "Try building without CUDA:"
    echo "  make clean && make -j"
    echo ""
    exit 1
fi
