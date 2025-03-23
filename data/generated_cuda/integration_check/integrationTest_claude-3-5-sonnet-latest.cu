#include <stdio.h>

__global__ void hello_kernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main() {
    int numThreads = 10;
    hello_kernel<<<1, numThreads>>>();
    cudaDeviceSynchronize();
    return 0;
}