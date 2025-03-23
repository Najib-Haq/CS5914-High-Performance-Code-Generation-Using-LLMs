#include <stdio.h>

// Kernel function to be executed by CUDA threads
__global__ void hello_cuda_kernel() {
    // Get the thread index
    int thread_idx = threadIdx.x;
    
    // Print the message from each thread
    printf("Hello from CUDA thread %d!\n", thread_idx);
}

int main() {
    // Define the number of threads to launch
    int num_threads = 10;
    
    // Launch the kernel
    hello_cuda_kernel<<<1, num_threads>>>();  // Block of num_threads threads, 1 block
    
    // Force the host to wait for the device to finish
    cudaDeviceSynchronize();
    
    return 0;
}