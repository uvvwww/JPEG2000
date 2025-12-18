#!/bin/bash
#SBATCH --account=ACD114118
#SBATCH --job-name=j2k_realtime
#SBATCH --partition=ctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=realtime_%j.log

echo "========================================="
echo "JPEG2000 Real Wall-Clock Time Test"
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
    
    rm -f output/photo2590_rt${threads}.j2k
    
    # 使用 /usr/bin/time 獲取真實時間
    /usr/bin/time -f "Real time: %e seconds\nUser time: %U seconds\nSystem time: %S seconds\nCPU usage: %P" \
        ./build/j2k_encode_pnm input/photo2590.ppm output/photo2590_rt${threads}.j2k 2>&1 | \
        grep -E "Using|Encoding time|Real time|User time|CPU usage"
    
    echo ""
done

echo "========================================="
echo "Summary: Check 'Real time' for actual speedup"
echo "========================================="
