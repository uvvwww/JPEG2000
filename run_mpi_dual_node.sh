#!/bin/bash
# MPI Multi-node Encoding Test Script
# Purpose: Test scalability across 2 nodes with 112 total cores

# Configuration
ACCOUNT="ACD114118"
NODES=2
CORES_PER_NODE=56
TOTAL_CORES=112

# MPI configuration options
# Option 1: 2 ranks (1 per node), each uses 56 threads
# Option 2: 4 ranks (2 per node), each uses 28 threads
# Option 3: 8 ranks (4 per node), each uses 14 threads

echo "========================================="
echo "MPI Dual-Node Encoding Performance Test"
echo "========================================="
echo "Total nodes: $NODES"
echo "Cores per node: $CORES_PER_NODE"
echo "Total cores: $TOTAL_CORES"
echo ""

# Prepare test images (you need multiple images for MPI ranks to process)
IMAGES=(
    "dataset/02.ppm"
    "dataset/03.ppm"
    # Add more images as needed
)

# Check if enough images exist
if [ ${#IMAGES[@]} -lt 2 ]; then
    echo "Warning: Only ${#IMAGES[@]} image(s) found. MPI works best with multiple images."
    echo "Creating dummy output paths for testing..."
fi

echo "========================================="
echo "Test 1: 2 MPI Ranks (1 per node, 56 threads each)"
echo "========================================="
export OMP_NUM_THREADS=56
srun -N $NODES -n 2 -c $CORES_PER_NODE -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "dataset/02.ppm" "output/node_test_1.j2k" \
    "dataset/03.ppm" "output/node_test_2.j2k"
echo ""

echo "========================================="
echo "Test 2: 4 MPI Ranks (2 per node, 28 threads each)"
echo "========================================="
export OMP_NUM_THREADS=28
srun -N $NODES -n 4 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "dataset/02.ppm" "output/node_test_1.j2k" \
    "dataset/03.ppm" "output/node_test_2.j2k" \
    "dataset/02.ppm" "output/node_test_3.j2k" \
    "dataset/03.ppm" "output/node_test_4.j2k"
echo ""

echo "========================================="
echo "Test 3: 8 MPI Ranks (4 per node, 14 threads each)"
echo "========================================="
export OMP_NUM_THREADS=14
srun -N $NODES -n 8 -c 14 -A $ACCOUNT \
    ./build/j2k_encode_mpi \
    "dataset/02.ppm" "output/node_test_1.j2k" \
    "dataset/03.ppm" "output/node_test_2.j2k" \
    "dataset/02.ppm" "output/node_test_3.j2k" \
    "dataset/03.ppm" "output/node_test_4.j2k" \
    "dataset/02.ppm" "output/node_test_5.j2k" \
    "dataset/03.ppm" "output/node_test_6.j2k" \
    "dataset/02.ppm" "output/node_test_7.j2k" \
    "dataset/03.ppm" "output/node_test_8.j2k"
echo ""

echo "========================================="
echo "All tests completed!"
echo "Check profiling_results.txt for detailed timing."
echo "========================================="
