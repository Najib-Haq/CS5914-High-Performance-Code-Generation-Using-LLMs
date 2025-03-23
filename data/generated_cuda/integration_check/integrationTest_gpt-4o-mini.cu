#include <iostream>

__global__ void helloFromCUDA() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from CUDA thread %d!\n", threadId);
}

int main() {
    const int numThreads = 16;
    const int numBlocks = 1;

    helloFromCUDA<<<numBlocks, numThreads>>>();
    cudaDeviceSynchronize();

    return 0;
}