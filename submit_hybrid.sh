#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --job-name=j2k_hybrid
#SBATCH --output=logs/j2k_hybrid_%j.log

# Load required modules
module load openmpi

# Set environment variables (4 ranks × 2 threads = 8 CPU cores)
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Create output directory if it doesn't exist
mkdir -p output logs

echo "======================================"
echo "Hybrid MPI+OpenMP JPEG2000 Encoding"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================"

# Run hybrid encoding
echo "Starting hybrid MPI+OpenMP encoding (4 ranks × 2 threads)..."
srun -n 4 --cpus-per-task=2 --mpi=pmix ./build/j2k_encode_mpi \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm

echo "======================================"
echo "Job completed"
ls -lh output/mpi_*.j2k 2>/dev/null | tail -5
echo "======================================"
