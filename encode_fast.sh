#!/bin/bash
# Fast JPEG2000 encoding with srun (no sbatch overhead)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image1.ppm> [image2.ppm ...]"
    echo "Example: $0 input/photo2590.ppm input/photo2590.ppm"
    exit 1
fi

module load openmpi

# Single rank, 8 OpenMP threads (fastest)
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "======================================"
echo "J2K Encoding with srun (1 rank × 8 threads)"
echo "======================================"
time srun --account=ACD114118 -n 1 --cpus-per-task=8 \
    ./build/j2k_encode_mpi "$@"
