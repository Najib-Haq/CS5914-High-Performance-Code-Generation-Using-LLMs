#include <stdio.h>
// #include <iostream>
#include <cuda_runtime.h>

// using namespace std;

__global__ void sum_reduction_serial(int *d_in, int *d_out, int N, int elements_per_thread) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate the start index for each thread
    int start_idx = tid * elements_per_thread;
    
    // Ensure that threads don't process beyond the array size
    if (start_idx < N) {
        int sum = 0;

        // Loop through the elements assigned to this thread
        for (int i = 0; i < elements_per_thread && start_idx + i < N; ++i) {
            sum += d_in[start_idx + i];
        }

        // Use atomicAdd to accumulate the sum safely
        atomicAdd(d_out, sum);
    }
}

int main() {
    const int N = 1024; // Number of elements
    const int elements_per_thread = 2; // Number of elements processed per thread
    int threads_per_block = 256;  // depends on hw?? TODO: need to know dynamic method
    int blocks_per_grid = (N + elements_per_thread * threads_per_block - 1) / (elements_per_thread * threads_per_block);


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
    sum_reduction_serial<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N, elements_per_thread);
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
