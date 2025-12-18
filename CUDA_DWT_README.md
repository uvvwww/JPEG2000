# CUDA DWT Acceleration for JPEG2000

This document describes the CUDA-accelerated DWT (Discrete Wavelet Transform) implementation for JPEG2000 encoding and decoding.

## Overview

The DWT operation accounts for approximately **4.1%** of the total encoding time according to profiling results. The CUDA implementation offloads these operations to the GPU while keeping other parts (T1, T2, MCT) running on CPU with OpenMP.

## Features

- **GPU-accelerated DWT**: Both forward and inverse transforms run on CUDA
- **5-3 Transform Support**: Reversible (lossless) wavelet transform
- **9-7 Transform Support**: Irreversible (lossy) wavelet transform  
- **Automatic Fallback**: Falls back to CPU implementation if CUDA fails
- **Compile-time Selection**: Enable/disable CUDA with a simple flag

## Building with CUDA Support

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 10.0 or later recommended)
- `nvcc` compiler in your PATH

### Compilation

**With CUDA enabled:**
```bash
make clean
make ENABLE_CUDA=1 -j
```

**Without CUDA (CPU-only, default):**
```bash
make clean
make -j
```

### Verify CUDA Build

When you run the encoder with CUDA enabled, you should see:
```
CUDA DWT initialized (Device 0)
```

If CUDA initialization fails, it will fall back to CPU automatically.

## Implementation Details

### Files Added

- `src/dwt_cuda.h` - CUDA DWT header with function declarations
- `src/dwt_cuda.cu` - CUDA implementation with kernels
- Modified `src/dwt.cpp` - Wrapper code to call CUDA or CPU versions
- Modified `Makefile` - CUDA compilation support

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                   │
│         (j2k_encode_pnm, j2k_decode_pnm)           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│             OpenJPEG Core (opj_encode)              │
│                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ DC Shift│  │   MCT   │  │   DWT   │◄──┐       │
│  │  (CPU)  │  │  (CPU)  │  │         │   │       │
│  └─────────┘  └─────────┘  └────┬────┘   │       │
│                                  │        │       │
│  ┌─────────┐  ┌─────────┐       │        │       │
│  │ T1 Quant│  │ T2 Stream│      │        │       │
│  │  (CPU)  │  │  (CPU)  │       │        │       │
│  └─────────┘  └─────────┘       │        │       │
└────────────────────────────────┬─┴────────┴───────┘
                                 │          │
                    ┌────────────▼──┐   ┌───▼──────┐
                    │  dwt.cpp      │   │dwt_cuda.cu│
                    │  (CPU/OpenMP) │   │ (GPU/CUDA)│
                    └───────────────┘   └───────────┘
```

### Function Mapping

| CPU Function (OpenMP)      | CUDA Function             | Transform Type |
|---------------------------|---------------------------|----------------|
| `opj_dwt_encode`          | `opj_dwt_encode_cuda`     | 5-3 Forward    |
| `opj_dwt_decode`          | `opj_dwt_decode_cuda`     | 5-3 Inverse    |
| `opj_dwt_encode_real`     | `opj_dwt_encode_real_cuda`| 9-7 Forward    |
| `opj_dwt_decode_real`     | `opj_dwt_decode_real_cuda`| 9-7 Inverse    |

### CUDA Kernel Design

- **Horizontal Pass**: Each thread block processes multiple rows in parallel
- **Vertical Pass**: Each thread block processes multiple columns in parallel
- **Shared Memory**: Used to cache row/column data for faster access
- **Memory Transfer**: Data copied to GPU at start of each tile, back to CPU at end

## Performance Considerations

### When CUDA Helps

- **Large Images**: >= 2K x 2K resolution
- **Multiple Resolution Levels**: More levels = more DWT work
- **Batch Processing**: Multiple images in sequence

### When CPU May Be Better

- **Small Images**: < 1K x 1K resolution (PCIe transfer overhead)
- **Single Image**: One-time encoding (CUDA initialization overhead)
- **Limited GPU Memory**: Very large tiles may not fit on GPU

### Profiling Results

Example with 57813×48438 image (4 CPU threads):

| Component    | Time (CPU) | Time (GPU)  | Speedup |
|-------------|-----------|-------------|---------|
| DWT         | 7.09 s    | ~2-3 s*     | ~2.5x   |
| T1 (quant)  | 116.95 s  | 116.95 s    | 1.0x    |
| **Total**   | **171.25 s**  | **~166-167 s** | **1.03x**|

*Estimated based on typical DWT GPU speedups. Actual results vary by GPU.

**Note**: T1 quantization dominates at 68.3% of total time. For maximum speedup, T1 should also be GPU-accelerated (future work).

## Troubleshooting

### CUDA Not Found

```
nvcc: command not found
```

**Solution**: Install CUDA Toolkit and add to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPU Out of Memory

```
CUDA error: out of memory
```

**Solution**: Reduce tile size or use CPU fallback automatically.

### No CUDA Device

```
No CUDA devices available
CUDA DWT decode failed, falling back to CPU
```

**Solution**: This is normal on systems without NVIDIA GPU. CPU version will be used automatically.

## Future Enhancements

1. **T1 Entropy Coding on GPU**: This is the bottleneck (68.3% of time)
2. **Optimized Memory Transfer**: Use CUDA streams for overlap
3. **Multi-GPU Support**: Distribute tiles across multiple GPUs
4. **Dynamic CPU/GPU Selection**: Choose based on image size
5. **Full 9-7 Lifting**: Current implementation is simplified

## Testing

### Basic Test (CPU-only)
```bash
make clean && make -j
./build/j2k_encode_pnm input/lena.ppm output/lena_cpu.j2k
./build/j2k_decode_pnm output/lena_cpu.j2k output/lena_cpu_decoded.ppm
```

### CUDA Test
```bash
make clean && make ENABLE_CUDA=1 -j
./build/j2k_encode_pnm input/lena.ppm output/lena_gpu.j2k
./build/j2k_decode_pnm output/lena_gpu.j2k output/lena_gpu_decoded.ppm
```

### Verify Results Match
```bash
diff output/lena_cpu.j2k output/lena_gpu.j2k
# Should be identical for 5-3 (reversible) transform
```

### Profile with CUDA
```bash
./build/j2k_encode_profile input/03.ppm output/03_cuda.j2k
# Look for "CUDA DWT initialized" message
```

## References

- OpenJPEG: https://github.com/uclouvain/openjpeg
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- JPEG2000 Standard (ISO/IEC 15444-1)

## License

Same as OpenJPEG (2-clause BSD license). See source files for details.
