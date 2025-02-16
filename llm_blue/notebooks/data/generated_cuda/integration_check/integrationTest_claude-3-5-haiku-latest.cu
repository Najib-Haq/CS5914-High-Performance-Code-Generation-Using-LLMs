#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", threadId);
}

int main() {
    int numThreads = 32;
    int numBlocks = 4;

    hello_kernel<<<numBlocks, numThreads>>>();

    // Synchronize to ensure kernel execution completes
    cudaDeviceSynchronize();

    // Check for any kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}