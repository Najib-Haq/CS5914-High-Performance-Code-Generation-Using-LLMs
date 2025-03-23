#include <stdio.h>

__global__ void helloKernel()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main()
{
    int numberOfThreads = 256;
    int numberOfBlocks = 1;

    helloKernel<<<numberOfBlocks, numberOfThreads>>>();
    cudaDeviceSynchronize();

    return 0;
}