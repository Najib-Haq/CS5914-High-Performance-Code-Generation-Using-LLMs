#include <iostream>

__global__ void hello_kernel() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", threadId);
}

int main() {
    int numThreads = 256;
    int numBlocks = 1;

    hello_kernel<<<numBlocks, numThreads>>>();

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    return 0;
}