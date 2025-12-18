# OpenJPEG Decode Acceleration Results

## Key Discovery: OPJ_NUM_THREADS Environment Variable

OpenJPEG has **built-in thread pool support** for decoding that was not being utilized!

## Solution

Set `OPJ_NUM_THREADS` environment variable to enable multi-threaded decode:
```bash
OPJ_NUM_THREADS=16 ./build/j2k_decode_profile input.j2k output.ppm
```

## Performance Results: 5184×3456 Image

### Before Optimization
- Single-thread (OMP_NUM_THREADS=1): **8.3755 s**
- Multi-thread (OMP_NUM_THREADS=16): 6.9631 s → 1.17x speedup ❌

### After Optimization (OPJ_NUM_THREADS)
- Single-thread (OPJ_NUM_THREADS=1): 8.3755 s (baseline)
- Multi-thread (OPJ_NUM_THREADS=16): **1.1526 s** → **7.26x speedup** ✅

## The Problem: Write Output Bottleneck

Even with 7.26x decode speedup, write output was taking **42.8% of total time**!

**Optimized Write Output**:
- Before: Per-pixel `fwrite()` calls (0.4937 s)
- After: Batch buffer + OpenMP parallel pixel conversion (0.0508 s) → **9.7x faster!**

## Final Performance: 5184×3456 with 16 Threads + Optimized Write

```
Decode (main):    0.6292 s (92.3%)
Write output:     0.0508 s (7.5%)
TOTAL TIME:       0.6816 s
```

**Overall speedup from original: 12.3x!**
- Original single-thread: ~8.4 s
- Optimized 16-thread: 0.68 s

## Comparison: Encode vs Decode Performance

| Operation | 1 core | 16 cores | Speedup |
|-----------|--------|----------|---------|
| **Encode** | 9.3665 s | 0.9635 s | 9.72x |
| **Decode** | 8.3755 s | 0.6816 s | 12.28x |

**Decode now matches or exceeds Encode speedup!**

## Key Implementation Details

### 1. Enable Thread Pool in Decode
OpenJPEG's `opj_t1_decode_cblks()` already has thread pool support:
- Checks `OPJ_NUM_THREADS` environment variable
- Submits individual codeblock decode jobs to thread pool
- Uses pthread-based worker threads

### 2. Batch I/O for Write Performance
Replaced per-pixel `fwrite()` with:
- Single buffer allocation for all pixels
- OpenMP parallel pixel processing loop
- Single batch `fwrite()` call

```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < pixels; i++) {
    // Convert pixel, write to buffer[i]
}
fwrite(buf, 1, pixels, fp);  // One system call!
```

### 3. Shift Constants Pre-calculated
Moved bit-shift calculations outside inner loops to avoid redundant computation.

## Recommendations

1. **Always set `OPJ_NUM_THREADS`** when decoding with OpenJPEG
   - Default: 0 (single-threaded)
   - Recommended: `OPJ_NUM_THREADS=ALL_CPUS` or actual thread count

2. **For file I/O**:
   - Use batch writes instead of per-item writes
   - Consider parallel pixel processing

3. **For MPI + OpenMP Hybrid**:
   - Set `OPJ_NUM_THREADS` per rank
   - Example: 4 MPI ranks × 4 OpenMP threads = 16 total

## Testing Commands

```bash
# Test with different thread counts
OPJ_NUM_THREADS=1 ./build/j2k_decode_profile input.j2k output.ppm
OPJ_NUM_THREADS=8 ./build/j2k_decode_profile input.j2k output.ppm
OPJ_NUM_THREADS=16 ./build/j2k_decode_profile input.j2k output.ppm
OPJ_NUM_THREADS=ALL_CPUS ./build/j2k_decode_profile input.j2k output.ppm

# With srun (SLURM)
srun -N 1 -c 16 bash -c "OPJ_NUM_THREADS=16 ./build/j2k_decode_profile input.j2k output.ppm"
```

## Conclusion

**Decode acceleration is absolutely possible with OpenJPEG!** The built-in thread pool support was just not enabled by default. With proper configuration and I/O optimization, decode can achieve **12x+ speedup on 16 cores**.
