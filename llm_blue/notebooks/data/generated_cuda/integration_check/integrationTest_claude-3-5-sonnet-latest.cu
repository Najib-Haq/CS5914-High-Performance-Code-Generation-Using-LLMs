#include <stdio.h>

__global__ void hello_kernel() {
    int tid = threadIdx.x;
    printf("Hello from CUDA thread %d!\n", tid);
}

int main() {
    hello_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}