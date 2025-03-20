#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>  // NVIDIA Management Library

#define THREADS_PER_BLOCK 1024

// Sequential addressing + Shared Memory Reduction Kernel

__global__ void reduceKernel(int* input, int* output, int size) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    if (i < size)
        shared_data[tid] = input[i] + (i + blockDim.x < size ? input[i + blockDim.x] : 0);
    else
        shared_data[tid] = 0;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = shared_data[0];
}

void reduce(int* d_input, int* d_output, int size) {
    int remaining = size;
    int blocks = (size + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);

    while (remaining > 1) {
        reduceKernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, remaining);
        cudaDeviceSynchronize();
        
        remaining = blocks;
        blocks = (remaining + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);

        std::swap(d_input, d_output);
    }

    cudaMemcpy(d_output, d_input, sizeof(int), cudaMemcpyDeviceToHost);
}

// Function to check GPU memory usage using NVML
void checkGpuMemoryUsage() {
    nvmlInit();
    nvmlDevice_t device;
    nvmlMemory_t memory;
    
    nvmlDeviceGetHandleByIndex(0, &device);  // Get first GPU
    nvmlDeviceGetMemoryInfo(device, &memory);

    std::cout << "GPU Memory Usage: " 
              << (memory.used / (1024 * 1024)) << " MB / "
              << (memory.total / (1024 * 1024)) << " MB" << std::endl;

    nvmlShutdown();
}

int main() {
    const int size = 1342177280; //1342177280; // 1.34 billion elements (~5GB)
    const size_t bytes = size * sizeof(int);

    int* h_input = new int[size];
    int h_output;

    for (int i = 0; i < size; i++) {
        h_input[i] = 1;
    }

    int* d_input, * d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    checkGpuMemoryUsage();  // GPU memory before copy

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    checkGpuMemoryUsage();  // GPU memory after copy

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(d_input, d_output, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Sum is " << h_output << std::endl;

    checkGpuMemoryUsage();  // GPU memory after computation

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
