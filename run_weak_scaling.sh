#!/bin/bash
# Weak Scaling Test: Keep workload per rank constant, increase total work
# Purpose: Test how well the system scales when adding more nodes/ranks

ACCOUNT="ACD114118"

echo "========================================="
echo "Weak Scaling Test"
echo "========================================="
echo "Each rank encodes 1 image with same thread count"
echo ""

# Test configuration: 1 image per rank, constant threads per rank
THREADS_PER_RANK=28
IMAGE_PER_RANK="dataset/02.ppm"

echo "----------------------------------------"
echo "Baseline: 1 node, 1 rank, 28 threads"
echo "----------------------------------------"
export OMP_NUM_THREADS=$THREADS_PER_RANK
srun -N 1 -n 1 -c $THREADS_PER_RANK -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMAGE_PER_RANK" "output/weak_1.j2k"
echo ""

echo "----------------------------------------"
echo "Scale 1: 1 node, 2 ranks, 28 threads each"
echo "----------------------------------------"
export OMP_NUM_THREADS=$THREADS_PER_RANK
srun -N 1 -n 2 -c $THREADS_PER_RANK -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMAGE_PER_RANK" "output/weak_1.j2k" \
    "$IMAGE_PER_RANK" "output/weak_2.j2k"
echo ""

echo "----------------------------------------"
echo "Scale 2: 2 nodes, 2 ranks, 28 threads each (1 rank per node)"
echo "----------------------------------------"
export OMP_NUM_THREADS=$THREADS_PER_RANK
srun -N 2 -n 2 -c $THREADS_PER_RANK -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMAGE_PER_RANK" "output/weak_1.j2k" \
    "$IMAGE_PER_RANK" "output/weak_2.j2k"
echo ""

echo "----------------------------------------"
echo "Scale 3: 2 nodes, 4 ranks, 28 threads each (2 ranks per node)"
echo "----------------------------------------"
export OMP_NUM_THREADS=$THREADS_PER_RANK
srun -N 2 -n 4 -c $THREADS_PER_RANK -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "$IMAGE_PER_RANK" "output/weak_1.j2k" \
    "$IMAGE_PER_RANK" "output/weak_2.j2k" \
    "$IMAGE_PER_RANK" "output/weak_3.j2k" \
    "$IMAGE_PER_RANK" "output/weak_4.j2k"
echo ""

echo "========================================="
echo "Weak scaling test completed!"
echo "Ideal: Execution time should remain constant"
echo "========================================="
