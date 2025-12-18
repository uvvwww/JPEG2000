#!/bin/bash

# Performance comparison script
echo "======================================"
echo "  JPEG2000 OPTIMIZATION COMPARISON"
echo "======================================"
echo ""

INPUT="input/photo2590.ppm"
OUTPUT_BASE="output/photo2590"

# Test different thread counts
for THREADS in 1 2 4 8; do
    echo "--- Testing with $THREADS OpenMP threads ---"
    export OMP_NUM_THREADS=$THREADS
    
    srun --account=ACD114118 -c $THREADS ./build/j2k_encode_profile \
        "$INPUT" "${OUTPUT_BASE}_${THREADS}t.j2k" 2>&1 | \
        grep -E "TOTAL TIME|T1 \(quant\)|DWT:" | head -3
    
    echo ""
done

echo "======================================"
echo "Results saved to profiling_results.txt"
echo "======================================"
