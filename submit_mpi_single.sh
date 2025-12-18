#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=j2k_mpi_single
#SBATCH --output=logs/j2k_mpi_single_%j.log

# Load required modules
module load openmpi

# Set environment variables
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Create output directory if it doesn't exist
mkdir -p output logs

# Print job info
echo "======================================"
echo "MPI JPEG2000 Encoding (Single Node)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "======================================"

# Run MPI encoding with 8 processes on 1 node
echo "Starting 8-process MPI encoding..."
srun -n 8 --mpi=pmix ./build/j2k_encode_mpi \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm \
    input/photo2590.ppm

echo "======================================"
echo "Job completed"
echo "Output files:"
ls -lh output/mpi_*.j2k 2>/dev/null | tail -10
echo "======================================"
