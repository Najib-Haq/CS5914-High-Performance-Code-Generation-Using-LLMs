#include <stdio.h>

__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main() {
    int threadsPerBlock = 256;
    int blocks = 1;

    helloKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}