cat logs/j2k_mpi_*.log
#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --partition=ctest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=j2k_mpi
#SBATCH --output=logs/j2k_mpi_%j.log

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
echo "MPI JPEG2000 Encoding Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "======================================"

# Prepare input images (create symbolic links to test data if needed)
# Example: assuming you have sample images in input/
# If you need to download or generate test images, do it here

# Run MPI encoding
# Example with 8 MPI processes (2 nodes × 4 processes) encoding 8 images
echo "Starting MPI encoding..."
srun -n $SLURM_NTASKS --mpi=pmix ./build/j2k_encode_mpi \
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
echo "Output files in: output/"
ls -lh output/mpi_*.j2k 2>/dev/null | tail -10
echo "======================================"
