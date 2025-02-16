#include <stdio.h>
// #include <iostream>
#include <cuda_runtime.h>

// using namespace std;

__global__ void sum_reduction_serial(int *input, int *output, int N) {
    __shared__ int shared_data[1024]; // Shared memory for block reduction
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < N) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0;
    }
    
    __syncthreads();
    
    // Serial reduction within the first thread of the block
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i];
        }
        atomicAdd(output, sum);
    }
}

int main() {
    const int N = 1024; // Number of elements
    int h_input[N], h_output = 0;
    int *d_input, *d_output;

    // benchmark
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = 1; // Example: all elements are 1
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with one block and 1024 threads (adjust if needed)
    cudaEventRecord(start);
    sum_reduction_serial<<<1, 1024>>>(d_input, d_output, N);
    cudaEventRecord(stop);

    // benchmark
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds); 

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Sum: %d\n", h_output);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
