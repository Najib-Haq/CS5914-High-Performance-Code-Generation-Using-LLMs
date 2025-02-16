#include <cuda_runtime.h>
#include <iostream>

__global__ void sumReductionKernel(int *d_input, int *d_output, int n) {
    __shared__ int sdata[512];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load shared memory
    sdata[tid] = (idx < n) ? d_input[idx] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate and initialize host memory
    int *h_input = new int[N];
    int *h_output = new int[NUM_BLOCKS];
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, NUM_BLOCKS * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch kernel
    sumReductionKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);

    // If more than one block was used, perform a second reduction
    if (NUM_BLOCKS > 1) {
        sumReductionKernel<<<1, BLOCK_SIZE>>>(d_output, d_output, NUM_BLOCKS);
    }

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Sum: " << h_output[0] << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}