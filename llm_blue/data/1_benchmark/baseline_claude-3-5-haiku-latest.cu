#include <cuda_runtime.h>
#include <iostream>

__global__ void sumReductionKernel(int *d_input, int *d_output) {
    __shared__ int sharedMem[1024];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedMem[tid] = d_input[index];
    __syncthreads();
    
    // Parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block-level sum to output
    if (tid == 0) {
        d_output[blockIdx.x] = sharedMem[0];
    }
}

int main() {
    const int ARRAY_SIZE = 1024;
    const int BLOCK_SIZE = 1024;
    const int GRID_SIZE = 1;

    // Host memory allocation and initialization
    int *h_input = new int[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = 1;
    }

    // Device memory allocation
    int *d_input, *d_output;
    cudaMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    cudaMalloc(&d_output, GRID_SIZE * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and measure time
    cudaEventRecord(start);
    sumReductionKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate kernel execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Final Sum: " << h_output << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}