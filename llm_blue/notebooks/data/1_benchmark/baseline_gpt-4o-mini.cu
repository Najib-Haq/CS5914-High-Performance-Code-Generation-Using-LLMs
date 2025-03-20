#include <cuda_runtime.h>
#include <iostream>

__global__ void sumReductionKernel(int *d_input, int *d_output, int n) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (globalIndex < n) {
        sharedData[tid] = d_input[globalIndex];
    } else {
        sharedData[tid] = 0; // Avoid reading out of bounds
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to the output
    if (tid == 0) {
        d_output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    const int ARRAY_SIZE = 1024;
    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int h_input[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = 1; // Initialize the array with 1s
    }

    int *d_input, *d_output, *h_output;
    h_output = new int[NUM_BLOCKS];

    cudaMalloc((void**)&d_input, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&d_output, NUM_BLOCKS * sizeof(int));

    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    sumReductionKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_input, d_output, ARRAY_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the intermediate results back to host
    cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on host
    int finalSum = 0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        finalSum += h_output[i];
    }

    std::cout << "Final Sum: " << finalSum << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Clean up
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}