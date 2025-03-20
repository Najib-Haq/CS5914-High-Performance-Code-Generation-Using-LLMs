#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main() {
    int num_threads = 256;
    helloKernel<<<1, num_threads>>>();
    cudaDeviceSynchronize();
    return 0;
}