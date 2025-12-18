#!/bin/bash
# Tile-based Parallel Encoding Test
# Purpose: Split single image across multiple MPI ranks (horizontal tiles)

ACCOUNT="ACD114118"

echo "========================================="
echo "Tile-Based Parallel Encoding Test"
echo "========================================="
echo "Mode: Single image split into horizontal tiles"
echo "Each MPI rank encodes one tile"
echo ""

INPUT="dataset/02.ppm"
OUTPUT_PREFIX="output/tile_test"

echo "----------------------------------------"
echo "Test 1: 1 node, 2 ranks (28 threads each)"
echo "Image split into 2 horizontal tiles"
echo "----------------------------------------"
export OMP_NUM_THREADS=28
srun -N 1 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "$INPUT" "$OUTPUT_PREFIX"
echo ""

echo "----------------------------------------"
echo "Test 2: 1 node, 4 ranks (14 threads each)"
echo "Image split into 4 horizontal tiles"
echo "----------------------------------------"
export OMP_NUM_THREADS=14
srun -N 1 -n 4 -c 14 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "$INPUT" "$OUTPUT_PREFIX"
echo ""

echo "----------------------------------------"
echo "Test 3: 1 node, 8 ranks (7 threads each)"
echo "Image split into 8 horizontal tiles"
echo "----------------------------------------"
export OMP_NUM_THREADS=7
srun -N 1 -n 8 -c 7 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "$INPUT" "$OUTPUT_PREFIX"
echo ""

echo "========================================="
echo "Tile parallel test completed!"
echo "Output files: ${OUTPUT_PREFIX}_tile0.j2k to _tileN.j2k"
echo "========================================="
