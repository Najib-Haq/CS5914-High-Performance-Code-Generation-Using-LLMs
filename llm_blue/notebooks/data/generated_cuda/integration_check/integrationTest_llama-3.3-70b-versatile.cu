#include <cstdio>

__global__ void printHelloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main() {
    int numBlocks = 1;
    int numThreads = 10;
    printHelloKernel<<<numBlocks, numThreads>>> ();
    cudaDeviceSynchronize();
    return 0;
}