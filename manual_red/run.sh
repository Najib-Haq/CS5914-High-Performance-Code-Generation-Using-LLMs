#!/bin/bash

# Set input sizes (1K, 1M, 1B, 2B)
INPUT_SIZES=(1024 1000000 1000000000 2000000000)

# CUDA source file and output binary name
CUDA_FILE_NAME="parallel_reduction4"
CUDA_FILE="${CUDA_FILE_NAME}.cu"
OUTPUT_BINARY="para.out"

# Compile CUDA code
echo "Compiling CUDA source file: $CUDA_FILE"
nvcc -o $OUTPUT_BINARY $CUDA_FILE -lnvidia-ml

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Loop through each input size and execute the program with profiling
for SIZE in "${INPUT_SIZES[@]}"; do
    echo "Running for input size: $SIZE"

    # Create a unique directory for storing results
    OUTPUT_DIR="${CUDA_FILE_NAME}_${SIZE}"
    mkdir -p $OUTPUT_DIR

    # Run the program and save output
    echo "Executing CUDA Program..."
    ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/output.txt"
    echo "Output saved to $OUTPUT_DIR/output.txt"

    # Perform profiling
    echo "Running NVIDIA Profiling..."

    # Basic profiling
    ncu ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/ncu_basic.txt"
    echo "Basic profiling saved to $OUTPUT_DIR/ncu_basic.txt"

    # Profiling with thread divergence benchmarks
    ncu --set full ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/ncu_threaddiv.txt"
    echo "Full profiling saved to $OUTPUT_DIR/ncu_threaddiv.txt"

    # Warp Execution Efficiency
    ncu --metrics smsp__thread_inst_executed_per_inst_executed ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/ncu_warp_efficiency.txt"
    echo "Warp execution efficiency saved to $OUTPUT_DIR/ncu_warp_efficiency.txt"

    # FLOPS (Simple Addition Operations)
    ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on,smsp__sass_thread_inst_executed_op_fmul_pred_on,smsp__sass_thread_inst_executed_op_ffma_pred_on ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/ncu_flops.txt"
    echo "FLOPS profiling saved to $OUTPUT_DIR/ncu_flops.txt"

    # Memory Usage Profiling
    ncu --metrics dram__bytes_read.sum,l1tex__t_bytes.sum,l2tex__t_bytes.sum ./$OUTPUT_BINARY $SIZE > "$OUTPUT_DIR/ncu_memory_usage.txt"
    echo "Memory usage profiling saved to $OUTPUT_DIR/ncu_memory_usage.txt"

    echo "Completed execution for input size: $SIZE"
done

echo "All profiling runs completed!"
