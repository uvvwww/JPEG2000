#!/bin/bash
# Debug MPI cross-node communication issues
# Test various MPI transport configurations

ACCOUNT="ACD114118"

echo "========================================="
echo "MPI Network Debug Tests"
echo "========================================="
echo ""

echo "Test 1: Force TCP transport (bypass InfiniBand)"
echo "----------------------------------------"
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
export OMP_NUM_THREADS=28
srun -N 2 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "dataset/02.ppm" "output/tcp_test"
echo ""

echo "Test 2: Disable UCX, use OB1 protocol"
echo "----------------------------------------"
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
export OMP_NUM_THREADS=28
srun -N 2 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "dataset/02.ppm" "output/ob1_test"
echo ""

echo "Test 3: Use available IB device (mlx5_0)"
echo "----------------------------------------"
export UCX_NET_DEVICES=mlx5_0:1
export OMP_NUM_THREADS=28
srun -N 2 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "dataset/02.ppm" "output/ib_test"
echo ""

echo "Test 4: TCP only (most compatible)"
echo "----------------------------------------"
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_btl_tcp_if_include=ib0
export OMP_NUM_THREADS=28
srun -N 2 -n 2 -c 28 -A $ACCOUNT \
    ./build/j2k_encode_mpi_tile \
    "dataset/02.ppm" "output/tcp_only"
echo ""

echo "========================================="
echo "Debug tests completed!"
echo "========================================="
