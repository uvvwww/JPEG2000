# JPEG2000 Encode vs Decode Performance Analysis

## Test Configuration: 5184×3456 Image

### Performance Results

#### Encoding
- **Single-thread (1 core)**: 9.3665 s
- **Multi-thread (16 cores)**: 0.9635 s
- **Speedup**: 9.72x

#### Decoding  
- **Single-thread (1 core)**: 8.1686 s
  - Decode main: 7.7602 s (95.0%)
  - Write output: 0.4018 s (4.9%)
- **Multi-thread (16 cores)**: 6.9631 s
  - Decode main: 6.5988 s (94.8%)
  - Write output: 0.3528 s (5.1%)
- **Speedup**: 1.17x

### Why Decode Has Poor Parallelization

#### 1. **Structural Differences**

**Encoding Pipeline (Naturally Parallel)**:
- DWT (正小波變換) - can process tiles independently
- T1 (量化) - can process codeblocks in parallel
- T2 (比特流生成) - can interleave parallel operations

**Decoding Pipeline (Sequential Dependencies)**:
- T2 (比特流解析) - must decode bitstream sequentially
- T1 (反量化) - decoder state dependencies between codeblocks
- DWT (逆小波變換) - requires all coefficients ready before reconstruction

#### 2. **Technical Constraints**

- **Entropy Decoding (T1)**: The MQ-coder (arithmetic coder) is inherently sequential
  - Each bit decoded depends on previous context and probability estimates
  - Cannot parallelize across codeblocks safely due to context state
  
- **Inverse DWT**: Requires coefficient reorganization 
  - Must wait for all detail/approximation coefficients
  - Dependencies between horizontal and vertical passes

#### 3. **Current OpenJPEG Status**

- OpenJPEG's `opj_decode()` is a single-threaded call
- No internal OpenMP pragmas in decode routines
- This is typical for JPEG2000 decoders (OpenJPEG, Kakadu, etc.)

### Speedup Comparison

| Configuration | Encode | Decode | Ratio |
|---|---|---|---|
| 1 core | 9.37 s | 8.17 s | 1.15x (encode slower) |
| 16 cores | 0.96 s | 6.96 s | 7.2x (encode faster) |

### Bottleneck Analysis

**Encode (1 core)**:
- T1 (quant): 94.8% of time
- Highly parallelizable in T1

**Decode (1 core)**:
- Decode main: 95.0% of time
- Mostly sequential T1 (inversion) and T2 (parsing)

### Recommendations for Decode Acceleration

1. **Tile-level parallelization** (already done with MPI)
   - Process different tiles on different processes
   - No data dependencies between tiles
   
2. **Custom decode implementation**
   - Rewrite T1 inversion to allow parallel coefficient processing
   - Optimize DWT with SIMD/OpenMP
   - Requires significant engineering

3. **Hardware acceleration**
   - GPU acceleration for DWT and coefficient processing
   - Beyond scope of CPU-only parallelization

## Conclusion

**Decode is fundamentally harder to parallelize than Encode** due to:
- Sequential bitstream decoding requirements
- Entropy coder state dependencies
- Data dependencies in coefficient reconstruction

With OpenJPEG, achieving 1.17x speedup with 16 threads on decode is reasonable.
For better decode performance, use **multi-tile / multi-process approaches** (MPI-based).
