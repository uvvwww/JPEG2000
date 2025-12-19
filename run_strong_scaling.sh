#!/bin/bash
# Strong Scaling Test: Fixed total workload, increase ranks
# Purpose: Measure speedup when adding more processors to same workload

ACCOUNT="ACD114118"

echo "========================================="
echo "Strong Scaling Test"
echo "========================================="
echo "Fixed workload: 4 images total"
echo "Increase MPI ranks to parallelize"
echo ""

# Fixed workload
IMG1="dataset/02.ppm"
IMG2="dataset/03.ppm"
OUT1="output/strong_1.j2k"
OUT2="output/strong_2.j2k"
OUT3="output/strong_3.j2k"
OUT4="output/strong_4.j2k"

echo "----------------------------------------"
echo "Baseline: 1 rank, 56 threads (1 node)"
echo "Processes 4 images sequentially"
echo "----------------------------------------"
export OMP_NUM_THREADS=56
srun -N 1 -n 1 -c 56 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMG1" "$OUT1" "$IMG2" "$OUT2" \
    "$IMG1" "$OUT3" "$IMG2" "$OUT4"
echo ""

echo "----------------------------------------"
echo "Scale 1: 2 ranks, 28 threads each (1 node)"
echo "Each rank processes 2 images"
echo "----------------------------------------"
export OMP_NUM_THREADS=28
srun -N 1 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMG1" "$OUT1" "$IMG2" "$OUT2" \
    "$IMG1" "$OUT3" "$IMG2" "$OUT4"
echo ""

echo "----------------------------------------"
echo "Scale 2: 4 ranks, 14 threads each (1 node)"
echo "Each rank processes 1 image"
echo "----------------------------------------"
export OMP_NUM_THREADS=14
srun -N 1 -n 4 -c 14 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMG1" "$OUT1" "$IMG2" "$OUT2" \
    "$IMG1" "$OUT3" "$IMG2" "$OUT4"
echo ""

echo "----------------------------------------"
echo "Scale 3: 4 ranks, 28 threads each (2 nodes)"
echo "Cross-node parallelism, each rank 1 image"
echo "----------------------------------------"
export OMP_NUM_THREADS=28
srun -N 2 -n 4 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMG1" "$OUT1" "$IMG2" "$OUT2" \
    "$IMG1" "$OUT3" "$IMG2" "$OUT4"
echo ""

echo "========================================="
echo "Strong scaling test completed!"
echo "Ideal: Execution time should decrease linearly with ranks"
echo "Speedup = T_baseline / T_parallel"
echo "========================================="
