#!/bin/bash
for level in 1 3 6; do
    echo "=== numresolution=$level ==="
    sed -i "s/parameters.numresolution = [0-9];/parameters.numresolution = $level;/" j2k_encode_pnm.cpp
    make -j4 > /dev/null 2>&1
    srun --account=ACD114118 -c 1 ./build/j2k_encode_pnm "input/sample_5184×3456.ppm" "output/sample_L${level}.j2k" 2>&1 | grep "Encoding time"
    ls -lh "output/sample_L${level}.j2k" | awk '{print "File size: " $5}'
    echo ""
done
