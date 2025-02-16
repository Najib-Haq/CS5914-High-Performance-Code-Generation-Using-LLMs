#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void sumReductionKernel(int *d_input, int *d_output) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < 1024) ? d_input[idx] : 0;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) d_output[blockIdx.x] = sdata[0];
}

int main() {
    int *h_input, *h_output = nullptr;
    int *d_input, *d_output;
    int size = 1024;
    size_t bytes = size * sizeof(int);
    cudaEvent_t start, stop;
    float milliseconds = 0;

    h_input = new int[size];
    h_output = new int[size / 1024]; // Assuming 1024 threads per block
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, size / 1024 * sizeof(int));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize input array on host
    for (int i = 0; i < size; i++) h_input[i] = 1;
    
    // Copy input from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Record start time
    cudaEventRecord(start);

    // Launch the kernel
    sumReductionKernel<<<1, 1024, 1024 * sizeof(int)>>>(d_input, d_output);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the result back to host and reduce the output
    cudaMemcpy(h_output, d_output, size / 1024 * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < size / 1024; ++i) total += h_output[i];

    // Calculate the time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the sum and the kernel execution time
    std::cout << "Sum: " << total << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_input;
    delete[] h_output;

    return 0;
}