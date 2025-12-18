#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --job-name=j2k_benchmark
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=benchmark_%j.log

echo "========================================="
echo "JPEG2000 Multi-core Performance Test"
echo "========================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Input: input/photo2590.ppm (6016x4000)"
echo ""

# 測試不同線程數
for threads in 1 2 4 8; do
    echo "========================================="
    echo "Testing with $threads threads"
    echo "========================================="
    
    export OMP_NUM_THREADS=$threads
    
    # 清除之前的輸出
    rm -f output/photo2590_t${threads}.j2k
    
    # 執行編碼
    ./build/j2k_encode_profile input/photo2590.ppm output/photo2590_t${threads}.j2k
    
    echo ""
done

echo "========================================="
echo "Benchmark completed at $(date)"
echo "========================================="
