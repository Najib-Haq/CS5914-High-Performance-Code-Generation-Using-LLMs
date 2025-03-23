#include <stdio.h>

__global__ void helloKernel() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", threadId);
}

int main() {
    int blockSize = 4;
    int numBlocks = 2;
    
    helloKernel<<<numBlocks, blockSize>>>();
    
    cudaError_t cudaResult = cudaDeviceSynchronize();
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaResult));
        return 1;
    }
    
    return 0;
}