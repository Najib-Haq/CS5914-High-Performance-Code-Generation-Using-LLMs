#include <stdio.h>

__global__ void helloFromCUDA() {
    int threadId = threadIdx.x;
    printf("Hello from CUDA thread %d!\n", threadId);
}

int main() {
    // Launch the kernel with 1 block and 10 threads
    helloFromCUDA<<<1, 10>>>();
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}