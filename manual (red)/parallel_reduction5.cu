#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>  // NVIDIA Management Library

#define THREADS_PER_BLOCK 1024

// Warp-Level Reduction Kernel
__inline__ __device__ int warpReduceSum(int sum) {
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

// Fully Optimized Reduction Kernel (Warp-Level)
__global__ void warpShuffleReduceKernel(int* input, int* output, long long int size) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned long long i = blockIdx.x * (blockDim.x * 2) + tid;

    // Load input safely and prevent out-of-bounds errors
    int sum = (i < size ? input[i] : 0) + (i + blockDim.x < size ? input[i + blockDim.x] : 0);
    shared_data[tid] = sum;
    __syncthreads();

    // Block-Level Reduction (Shared Memory)
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Warp-Level Reduction (Using Shuffle Instructions)
    if (tid < 32) {
        int val = shared_data[tid];
        // warp shuffle reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) output[blockIdx.x] = val;
    }
}

// Reduce Partial Sums on CPU
int finalReduceOnCPU(int* d_output, int numBlocks) {
    int* h_partial_sums = new int[numBlocks];
    cudaMemcpy(h_partial_sums, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_partial_sums[i];
    }

    delete[] h_partial_sums;
    return totalSum;
}

// GPU Memory Check Function
void checkGpuMemoryUsage(const char* message) {
    nvmlInit();
    nvmlDevice_t device;
    nvmlMemory_t memory;
    nvmlDeviceGetHandleByIndex(0, &device);
    nvmlDeviceGetMemoryInfo(device, &memory);
    std::cout << message << " - GPU Memory Usage: " 
              << (memory.used / (1024 * 1024)) << " MB / "
              << (memory.total / (1024 * 1024)) << " MB\n";
    nvmlShutdown();
}

// Host Function to Call Kernel
void reduce(int* d_input, int* d_output, long long int size) {
    int numBlocks = (size + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    warpShuffleReduceKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int finalSum = finalReduceOnCPU(d_output, numBlocks);
    printf("Warp Shuffle Reduction Kernel Execution Time: %f ms\n", milliseconds);
    printf("Correct Final Sum: %d\n", finalSum);
}

// Main Function
int main() {
    const long long int size = 1024; // Adjustable (1K, 1M, 1B, 5B)
    const size_t bytes = size * sizeof(int);
    
    int* h_input = new int[size];
    for (long long int i = 0; i < size; i++) h_input[i] = 1;

    int *d_input, *d_output;
    checkGpuMemoryUsage("Before GPU Memory Allocation");

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, ((size + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2)) * sizeof(int));  

    checkGpuMemoryUsage("After GPU Memory Allocation");

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    checkGpuMemoryUsage("After Copying Data to GPU");

    reduce(d_input, d_output, size);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    checkGpuMemoryUsage("After GPU Memory Deallocation");
    return 0;
}
