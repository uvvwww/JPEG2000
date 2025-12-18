# CUDA DWT Tuning Guide

This document explains how to tune CUDA DWT performance parameters for experimentation.

## Quick Start

Edit [src/dwt_cuda.cu](src/dwt_cuda.cu) lines 8-48 or pass these as compiler flags in the Makefile.

## Tunable Parameters

### 1. **DWT_THREADS_PER_BLOCK** (Default: 256)
**What it controls**: Number of CUDA threads per block  
**Impact**: GPU occupancy, register usage, shared memory

**Recommended values to test**:
- `64` - Low occupancy, may underutilize GPU
- `128` - Moderate occupancy
- `256` - **DEFAULT** - Good balance for most GPUs
- `512` - Higher occupancy, more register pressure
- `1024` - Maximum threads (may cause spill to local memory)

**How to change**:
```bash
# In Makefile, add to NVCC_FLAGS:
NVCC_FLAGS += -DDWT_THREADS_PER_BLOCK=512

# Or edit dwt_cuda.cu line 31:
#define DWT_THREADS_PER_BLOCK 512
```

**Expected behavior**:
- Too low (64): GPU underutilized, slower
- Too high (1024): Register spill, slower
- Sweet spot varies by GPU architecture (Pascal/Volta/Ampere)

---

### 2. **DWT_USE_ASYNC_MEMCPY** (Default: 1)
**What it controls**: Whether memory transfers overlap with computation  
**Impact**: PCIe bus utilization

**Values**:
- `0` = Synchronous (easier to debug, profile)
- `1` = **DEFAULT** - Asynchronous (overlap transfers)

**How to change**:
```bash
# Disable for profiling:
NVCC_FLAGS += -DDWT_USE_ASYNC_MEMCPY=0
```

**Expected behavior**:
- `0`: Clearer profiling timelines, ~5-10% slower
- `1`: Better throughput but harder to profile

---

### 3. **DWT_PERSISTENT_BUFFER_LIMIT** (Default: 512MB)
**What it controls**: Maximum GPU memory to cache between operations  
**Impact**: Memory allocation overhead vs GPU memory usage

**Recommended values**:
- `0` - No caching (allocate every time)
- `268435456` - 256MB cache
- `536870912` - **DEFAULT** 512MB cache
- `-1` - Unlimited (cache everything)

**How to change**:
```bash
# Cache everything (fast but uses GPU memory):
NVCC_FLAGS += -DDWT_PERSISTENT_BUFFER_LIMIT=-1

# No caching (slow but minimal GPU memory):
NVCC_FLAGS += -DDWT_PERSISTENT_BUFFER_LIMIT=0
```

**Expected behavior**:
- `0`: +500ms overhead per image (6 malloc/free cycles)
- `-1`: Fastest, but may run out of GPU memory on huge images
- For your 57813×37145 image = 8GB, so 512MB cache won't help much

---

### 4. **DWT_UNROLL_FACTOR** (Default: 4)
**What it controls**: Loop unrolling in lifting kernels  
**Impact**: Instruction-level parallelism vs register pressure

**NOTE**: Currently not actively used in code (hardcoded to 4 in pragmas). This parameter is reserved for future experimentation. To change unroll factor, you must manually edit the `#pragma unroll 4` directives in [src/dwt_cuda.cu](src/dwt_cuda.cu).

**Recommended values**:
- `1` - No unrolling
- `2` - Moderate
- `4` - **DEFAULT** (hardcoded)
- `8` - Aggressive (may cause register spill)

**How to change**:
```bash
# Currently requires manual code edit (not macro-controlled)
# Search for "#pragma unroll 4" in dwt_cuda.cu and change to desired value
```

**Expected behavior**:
- Higher = More ILP, but increases register usage
- Diminishing returns beyond 4 for most cases

---

### 5. **DWT_DEBUG_PRINT** (Default: 0)
**What it controls**: Print kernel launch parameters  
**Impact**: Debugging visibility (adds print overhead)

**Values**:
- `0` - **DEFAULT** - Silent
- `1` - Print block/thread counts

**How to change**:
```bash
NVCC_FLAGS += -DDWT_DEBUG_PRINT=1
```

**Output example**:
```
[DWT] Level 5: V-pass blocks=227 threads=256
[DWT] Level 5: H-pass blocks=145 threads=256
[DWT] Decode res=3: H-pass blocks=145 threads=256
```

---

## Recommended Experiments

### Experiment 1: Find Optimal Block Size
```bash
for THREADS in 64 128 256 512 1024; do
  make clean
  make ENABLE_CUDA=1 NVCC_FLAGS="-DDWT_THREADS_PER_BLOCK=$THREADS" -j32
  echo "Testing $THREADS threads:"
  ./build/j2k_encode_pnm input/photo2590.ppm output/test.j2k | grep "DWT:"
done
```

### Experiment 2: Test Persistent Caching Impact
```bash
# No cache
make clean && make ENABLE_CUDA=1 NVCC_FLAGS="-DDWT_PERSISTENT_BUFFER_LIMIT=0" -j32

# Unlimited cache  
make clean && make ENABLE_CUDA=1 NVCC_FLAGS="-DDWT_PERSISTENT_BUFFER_LIMIT=-1" -j32
```

### Experiment 3: Profile with Synchronous Transfers
```bash
make clean
make ENABLE_CUDA=1 NVCC_FLAGS="-DDWT_USE_ASYNC_MEMCPY=0" -j32
nvprof --print-gpu-trace ./build/j2k_encode_pnm input/sample.ppm output/test.j2k
```

---

## Understanding Your Bottleneck

For 57813×37145 image (8.6GB):
- **PCIe transfer**: 51.6GB ÷ 25 GB/s = **~2 seconds minimum**
- **Actual DWT**: Currently 18.9s (12-15s transfer + 3-6s compute)

**Key insight**: Your current 18.9s DWT time suggests transfer overhead dominates. Tuning block size or unrolling will only improve the 3-6s compute portion, not the 12-15s transfer portion.

**Best tuning strategy**:
1. Test `DWT_PERSISTENT_BUFFER_LIMIT=-1` (reduces malloc overhead)
2. Test block sizes 256/512 (may reduce compute from 6s to 4s)
3. Consider processing all 3 components in one GPU call (requires deeper code changes)

---

## Profiling Commands

```bash
# See detailed GPU timeline
nvprof --print-gpu-trace ./build/j2k_encode_pnm input/photo2590.ppm output/test.j2k

# Check memory transfer times
nvprof --print-api-trace ./build/j2k_encode_pnm input/photo2590.ppm output/test.j2k | grep Memcpy

# Check kernel execution times
nvprof --kernels dwt53 --print-gpu-summary ./build/j2k_encode_pnm input/photo2590.ppm output/test.j2k
```
