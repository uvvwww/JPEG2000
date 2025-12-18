#!/bin/bash
for level in 1 2 3 4 5 6; do
    echo "=== Testing numresolution=$level ==="
    # 暫時修改參數
    sed -i "s/parameters.numresolution = [0-9];/parameters.numresolution = $level;/" j2k_encode_pnm.cpp
    make j2k_encode_pnm 2>&1 | grep -E "g\+\+.*j2k_encode_pnm" || true
    srun --account=ACD114118 -c 1 ./build/j2k_encode_pnm input/lena.ppm output/lena_L${level}.j2k 2>&1 | grep "Encoding time"
    ls -lh output/lena_L${level}.j2k | awk '{print "File size: " $5}'
done
