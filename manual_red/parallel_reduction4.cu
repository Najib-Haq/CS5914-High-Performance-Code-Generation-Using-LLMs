#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>  // NVIDIA Management Library

#define THREADS_PER_BLOCK 1024

// Unrolling the Final Warp Loop

// Warp-Level Reduction Using Template Metaprogramming for Full Unrolling
template <int Offset>
__inline__ __device__ void warpReduceSumUnrolled(int &sum) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, Offset);
}


// Fully Unrolled Block-Level Reduction Using Template Metaprogramming
template <int BlockSize>
__inline__ __device__ void blockReduceSum(int* shared_data, int tid) {
    if constexpr (BlockSize >= 1024) {
        if (tid < 512) shared_data[tid] += shared_data[tid + 512];
        __syncthreads();
    }
    if constexpr (BlockSize >= 512) {
        if (tid < 256) shared_data[tid] += shared_data[tid + 256];
        __syncthreads();
    }
    if constexpr (BlockSize >= 256) {
        if (tid < 128) shared_data[tid] += shared_data[tid + 128];
        __syncthreads();
    }
    if constexpr (BlockSize >= 128) {
        if (tid < 64) shared_data[tid] += shared_data[tid + 64];
        __syncthreads();
    }
    if (tid <= 32) {
        warpReduceSumUnrolled<32>(shared_data[tid]);  // changed and added this
        warpReduceSumUnrolled<16>(shared_data[tid]);
        warpReduceSumUnrolled<8>(shared_data[tid]);
        warpReduceSumUnrolled<4>(shared_data[tid]);
        warpReduceSumUnrolled<2>(shared_data[tid]);
        warpReduceSumUnrolled<1>(shared_data[tid]);
    }
}

// Fully Unrolled CUDA Reduction Kernel
__global__ void fullyUnrolledReduceKernel(int* input, int* output, long long int size) {
    __shared__ int shared_data[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned long long i = blockIdx.x * (blockDim.x * 2) + tid;

    // Load input safely and prevent skipping values
    int sum = 0;
    if (i < size) sum = input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];

    shared_data[tid] = sum;
    __syncthreads();

    // Perform Fully Unrolled Block Reduction
    blockReduceSum<THREADS_PER_BLOCK>(shared_data, tid);

    // Store only final block sum
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
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
    fullyUnrolledReduceKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int finalSum = finalReduceOnCPU(d_output, numBlocks);
    printf("Fully Unrolled Reduction Kernel Execution Time: %f ms\n", milliseconds);
    printf("Correct Final Sum: %d\n", finalSum);
}

// Main Function
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Wrong Usage: " << argv[0] << " <size>\n";
        return 1;
    }
    const long long int size = atoll(argv[1]); //1342177280; //1342177280; // 1.34 billion elements (~5GB)
    if (size <= 0) {
        std::cerr << "Error: Invalid input size.\n";
        return 1;
    }
    std::cout << "Running CUDA Reduction for size: " << size << std::endl;
    const size_t bytes = size * sizeof(int);
    
    int* h_input = new int[size];
    for (long long int i = 0; i < size; i++) h_input[i] = 1;

    int *d_input, *d_output;
    checkGpuMemoryUsage("Before GPU Memory Allocation");

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, ((size + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2)) * sizeof(int));  // Fix: Allocate full output array for all blocks

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
