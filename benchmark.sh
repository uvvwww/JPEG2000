#!/bin/bash

# JPEG2000 MPI Encoder Performance Benchmark
# Tests multiple configurations across different dataset sizes

ACCOUNT="ACD114118"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="benchmark_results_${TIMESTAMP}.csv"
LOG_FILE="benchmark_${TIMESTAMP}.log"

# MPI environment settings
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
export OMPI_MCA_coll=^hcoll

# Datasets
SMALL="dataset/01.ppm"
MEDIUM="dataset/02.ppm"

# Header
{
    echo "=========================================="
    echo "JPEG2000 MPI Encoder Performance Benchmark"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Account: $ACCOUNT"
    echo ""
} | tee "$LOG_FILE"

# CSV Header
echo "config,dataset,size_MB,total_time_sec" > "$CSV_FILE"

run_test() {
    local config_name=$1
    local dataset=$2
    local dataset_label=$3
    local cmd=$4
    
    if [ ! -f "$dataset" ]; then
        echo "[SKIP] $config_name | $dataset_label | file not found" | tee -a "$LOG_FILE"
        echo "$config_name,$dataset_label,ERROR,N/A" >> "$CSV_FILE"
        return
    fi
    
    local size_mb=$(du -m "$dataset" | cut -f1)
    
    echo -n "Testing $config_name on $dataset_label ($size_mb MB)... " | tee -a "$LOG_FILE"
    
    # Run command and measure wall time
    local start_ns=$(date +%s%N)
    eval "$cmd" > /tmp/bench_output.txt 2>&1
    local exit_code=$?
    local end_ns=$(date +%s%N)
    
    local elapsed_ns=$((end_ns - start_ns))
    local elapsed_sec=$(echo "scale=4; $elapsed_ns / 1000000000" | bc)
    
    if [ $exit_code -eq 0 ]; then
        echo "${elapsed_sec}s" | tee -a "$LOG_FILE"
        echo "$config_name,$dataset_label,$size_mb,$elapsed_sec" >> "$CSV_FILE"
    else
        echo "FAILED (exit $exit_code)" | tee -a "$LOG_FILE"
        echo "$config_name,$dataset_label,$size_mb,FAILED" >> "$CSV_FILE"
        cat /tmp/bench_output.txt >> "$LOG_FILE"
    fi
}

echo "" | tee -a "$LOG_FILE"
echo "=== Config 1: baseline (1 core) ===" | tee -a "$LOG_FILE"
run_test "baseline_1c" "$SMALL" "small" \
    "srun --account=$ACCOUNT -N 1 -n 1 -c 1 ./build/j2k_encode_pnm $SMALL /tmp/out.j2k"
run_test "baseline_1c" "$MEDIUM" "medium" \
    "srun --account=$ACCOUNT -N 1 -n 1 -c 1 ./build/j2k_encode_pnm $MEDIUM /tmp/out.j2k"

echo "" | tee -a "$LOG_FILE"
echo "=== Config 2: OpenMPI 1N 1n 32c (1 node, 1 MPI, 32 cores) ===" | tee -a "$LOG_FILE"
run_test "1N_1n_32c" "$SMALL" "small" \
    "srun --account=$ACCOUNT -N 1 -n 1 -c 32 ./build/j2k_encode_pnm $SMALL /tmp/out.j2k"
run_test "1N_1n_32c" "$MEDIUM" "medium" \
    "srun --account=$ACCOUNT -N 1 -n 1 -c 32 ./build/j2k_encode_pnm $MEDIUM /tmp/out.j2k"

echo "" | tee -a "$LOG_FILE"
echo "=== Config 3: OpenMPI 1N 2n 16c (1 node, 2 MPI, 16 cores each) ===" | tee -a "$LOG_FILE"
run_test "1N_2n_16c" "$SMALL" "small" \
    "srun --account=$ACCOUNT -N 1 -n 2 -c 16 ./build/j2k_encode_mpi_tile $SMALL /tmp/out"
run_test "1N_2n_16c" "$MEDIUM" "medium" \
    "srun --account=$ACCOUNT -N 1 -n 2 -c 16 ./build/j2k_encode_mpi_tile $MEDIUM /tmp/out"

echo "" | tee -a "$LOG_FILE"
echo "=== Config 4: OpenMPI 2N 2n 16c (2 nodes, 2 MPI, 16 cores each) ===" | tee -a "$LOG_FILE"
run_test "2N_2n_16c" "$SMALL" "small" \
    "srun --account=$ACCOUNT -N 2 -n 2 -c 16 ./build/j2k_encode_mpi_tile $SMALL /tmp/out"
run_test "2N_2n_16c" "$MEDIUM" "medium" \
    "srun --account=$ACCOUNT -N 2 -n 2 -c 16 ./build/j2k_encode_mpi_tile $MEDIUM /tmp/out"

echo "" | tee -a "$LOG_FILE"
echo "=== Config 5: OpenMPI 2N 4n 8c (2 nodes, 4 MPI, 8 cores each) ===" | tee -a "$LOG_FILE"
run_test "2N_4n_8c" "$SMALL" "small" \
    "srun --account=$ACCOUNT -N 2 -n 4 -c 8 ./build/j2k_encode_mpi_tile $SMALL /tmp/out"
run_test "2N_4n_8c" "$MEDIUM" "medium" \
    "srun --account=$ACCOUNT -N 2 -n 4 -c 8 ./build/j2k_encode_mpi_tile $MEDIUM /tmp/out"

# Summary
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Benchmark completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results:" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
column -t -s',' "$CSV_FILE" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "CSV file: $CSV_FILE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
