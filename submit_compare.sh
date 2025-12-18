#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=compare_16cores
#SBATCH --output=logs/compare_%j.log

module load openmpi

mkdir -p output logs

export OMP_NUM_THREADS=4

echo "======================================"
echo "方案B: 4 ranks × 4 threads (16核並行)"
echo "======================================"
srun ./build/j2k_encode_mpi \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm

echo "======================================"
