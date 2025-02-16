#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d!\n", idx);
}

int main() {
    // Define number of blocks and threads per block
    int blocks = 1;
    int threadsPerBlock = 10;

    // Launch the kernel
    helloKernel<<<blocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}