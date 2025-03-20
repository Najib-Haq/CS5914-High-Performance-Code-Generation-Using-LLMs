#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void FixDivergenceKernel(int* input, int* output) {
    unsigned int i = threadIdx.x; //threads start next to each other
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) { // furthest element is blockDim away
        if (threadIdx.x < stride) { // 
            input[i] += input[i + stride]; // each thread adds a distant element to its assigned position
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
    *output = input[0];
    }
}

int main() {
    // Size of the input data
    const int size = 1024;
    const int bytes = size * sizeof(int);

    // Allocate memory for input and output on host
    int* h_input = new int[size];
    int* h_output = new int;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(int));

    // benchmark
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Launch the kernel
    FixDivergenceKernel<<<1, size / 2>>>(d_input, d_output);
    cudaEventRecord(stop);

    // benchmark
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds); 

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum is " << *h_output << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}