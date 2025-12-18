#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --job-name=j2k_single
#SBATCH --output=logs/j2k_single_%j.log

# Load required modules
module load openmpi

# Single rank, 8 threads
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close

mkdir -p output logs

echo "======================================"
echo "Single Rank + 8 OpenMP Threads"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================"

# Run with 1 MPI rank, 8 OpenMP threads
srun -n 1 --cpus-per-task=8 --mpi=pmix ./build/j2k_encode_mpi \
    input/photo2590.ppm

echo "======================================"
echo "Job completed"
ls -lh output/mpi_*.j2k 2>/dev/null | tail -3
echo "======================================"
