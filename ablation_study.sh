#!/bin/bash

# Ablation Study for JPEG2000 Encoder Optimizations
# Tests: baseline, thread_pool/T1, IO/buffer, Pixel_parallel

ACCOUNT="ACD114118"
DATASET="dataset/03.ppm"  # Large image for meaningful results
OUTPUT_PREFIX="/tmp/ablation"

echo "=========================================="
echo "JPEG2000 Encoder Ablation Study"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Date: $(date)"
echo ""

# Function to run profiling test
run_profile_test() {
    local config_name=$1
    local num_threads=$2
    
    echo ""
    echo "=== $config_name ==="
    
    export OMP_NUM_THREADS=$num_threads
    
    srun --account=$ACCOUNT -N 1 -n 1 -c $num_threads \
        ./build/j2k_encode_profile $DATASET ${OUTPUT_PREFIX}_${config_name}.j2k 2>&1 | \
        grep -E "(DWT|T1|Load image|TOTAL TIME|DC shift)"
}

echo ""
echo "=========================================="
echo "Test 1: Baseline (1 core)"
echo "=========================================="
run_profile_test "baseline_1c" 1

echo ""
echo "=========================================="
echo "Test 2: thread_pool/T1_parallel (16 cores)"
echo "=========================================="
run_profile_test "t1_parallel_16c" 16

echo ""
echo "=========================================="
echo "Test 3: All optimizations (32 cores)"
echo "=========================================="
run_profile_test "full_opt_32c" 32

echo ""
echo "=========================================="
echo "Ablation Study Complete"
echo "=========================================="
