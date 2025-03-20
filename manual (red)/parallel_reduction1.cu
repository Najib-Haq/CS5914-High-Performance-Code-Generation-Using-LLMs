#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>



// Simple parallel
// TODO: Update to input size of millions & billions



__global__ void SimpleSumReductionKernel(float* input, float* output) {
    
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {

            // take the input every stride apart and add (stride = 1, 2, 4, 8, ..)
            input[i] += input[i + stride];
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

int main() {
    // Size of the input data
    int size = 1024;
    int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f; // Example: Initialize all elements to 1
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // benchmark
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    // Launch the kernel
    SimpleSumReductionKernel<<<1, size / 2>>>(d_input, d_output);
    cudaEventRecord(stop);

    // benchmark
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds); 

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum is " << *h_output << std::endl;

    // Calculate the optimal block size - will need later
    // int blockSize = 1024; // Optimal block size for many devices
    // int minGridSize = 40;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, SimpleSumReductionKernel, 0, 0);


    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}