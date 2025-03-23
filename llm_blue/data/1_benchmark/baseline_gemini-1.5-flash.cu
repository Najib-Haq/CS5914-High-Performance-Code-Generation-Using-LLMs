#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Define the block size for the reduction
#define BLOCK_SIZE 256

__global__ void sumReductionKernel(int *d_input, int *d_output, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    // Load data into shared memory
    if (i < n) {
        sum = d_input[i];
    }

    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = sum;
    }
}


int main() {
    int n = 1024;
    int *h_input, *h_output, *d_input, *d_output;
    cudaEvent_t start, stop;

    // Allocate host memory
    h_input = new int[n];
    h_output = new int[ (n + BLOCK_SIZE -1) / BLOCK_SIZE];

    // Initialize host input array
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1;
    }


    // Allocate device memory
    cudaMalloc((void **)&d_input, n * sizeof(int));
    cudaMalloc((void **)&d_output, ((n + BLOCK_SIZE -1) / BLOCK_SIZE) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    int blocks = (n + BLOCK_SIZE -1) / BLOCK_SIZE;
    sumReductionKernel<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);

    // Record stop event
    cudaEventRecord(stop, 0);

    // Synchronize
    cudaEventSynchronize(stop);


    // Copy the final sum back to the host
    cudaMemcpy(h_output, d_output, ((n + BLOCK_SIZE -1) / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    int finalSum = 0;
    for(int i = 0; i < blocks; ++i){
        finalSum += h_output[i];
    }

    // Measure kernel execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the sum and the kernel execution time
    std::cout << "Sum: " << finalSum << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;

    return 0;
}