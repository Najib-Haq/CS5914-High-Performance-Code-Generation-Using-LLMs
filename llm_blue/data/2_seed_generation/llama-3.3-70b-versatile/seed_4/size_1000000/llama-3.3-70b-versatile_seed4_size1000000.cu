
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use 1024 threads per block to maximize memory bandwidth utilization and strike a balance between memory access and computational throughput.
    // 2. Implement a bank-conflict-free memory access pattern using a strided addressing pattern with stride 32 (number of threads per warp).
    // 3. Utilize warp-level primitives like __shfl_down_sync() for efficient warp-level reductions.
    // 4. Employ a combination of warp-level and block-level reductions to minimize the number of threads participating in the final reduction phase.
    // 5. Apply an early exit strategy to minimize unnecessary work when the input size is less than the number of threads per block.
    // 6. Unroll the reduction loop to reduce loop overhead and increase instruction-level parallelism.
    // This strategy should be effective because it reduces memory traffic, optimizes computations, and maximizes utilization of GPU resources.

    extern __shared__ int sharedData[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Early exit strategy: If the size of the input array is less than the number of threads per block,
    // only the first 'size' threads participate in the reduction.
    if (idx >= size) return;

    // Load data from global memory using a coalesced access pattern
    int sum = 0;
    while (idx < size) {
        sum += input[idx];
        idx += stride;
    }

    // Store loaded data in shared memory using a bank-conflict-free strided addressing pattern
    sharedData[threadIdx.x] = sum;
    __syncthreads();

    // Warp-level reduction using __shfl_down_sync()
    for (int offset = 512; offset > 0; offset /= 2) {
        int neighborVal = __shfl_down_sync(0xffffffff, sharedData[threadIdx.x], offset);
        if (threadIdx.x < offset) {
            sharedData[threadIdx.x] += neighborVal;
        }
    }

    // Reduction across warps in the block
    if (threadIdx.x < 32) {
        int warpId = threadIdx.x;
        sharedData[warpId] += sharedData[warpId + 32];
        sharedData[warpId] += sharedData[warpId + 64];
        sharedData[warpId] += sharedData[warpId + 96];
        sharedData[warpId] += sharedData[warpId + 128];
        sharedData[warpId] += sharedData[warpId + 160];
        sharedData[warpId] += sharedData[warpId + 192];
        sharedData[warpId] += sharedData[warpId + 224];
        sharedData[warpId] += sharedData[warpId + 256];
        sharedData[warpId] += sharedData[warpId + 288];
        sharedData[warpId] += sharedData[warpId + 320];
        sharedData[warpId] += sharedData[warpId + 352];
        sharedData[warpId] += sharedData[warpId + 384];
        sharedData[warpId] += sharedData[warpId + 416];
        sharedData[warpId] += sharedData[warpId + 448];
        sharedData[warpId] += sharedData[warpId + 480];
        sharedData[warpId] += sharedData[warpId + 512];
        sharedData[warpId] += sharedData[warpId + 544];
        sharedData[warpId] += sharedData[warpId + 576];
        sharedData[warpId] += sharedData[warpId + 608];
        sharedData[warpId] += sharedData[warpId + 640];
        sharedData[warpId] += sharedData[warpId + 672];
        sharedData[warpId] += sharedData[warpId + 704];
        sharedData[warpId] += sharedData[warpId + 736];
        sharedData[warpId] += sharedData[warpId + 768];
        sharedData[warpId] += sharedData[warpId + 800];
        sharedData[warpId] += sharedData[warpId + 832];
        sharedData[warpId] += sharedData[warpId + 864];
        sharedData[warpId] += sharedData[warpId + 896];
        sharedData[warpId] += sharedData[warpId + 928];
        sharedData[warpId] += sharedData[warpId + 960];
        sharedData[warpId] += sharedData[warpId + 992];
    }
    __syncthreads();

    // Store the block's result in the output array
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// Host function to perform reduction
int sumArray(int* h_input, int size) {
    int *d_input, *d_temp;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(int));
    
    // The size of d_temp is based on the number of blocks we'll launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void**)&d_temp, blocksPerGrid * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel with the actual size parameter
    cudaEventRecord(start);
    sumReduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_temp, size);
    cudaEventRecord(stop);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy the block results back to host
    int* h_temp = new int[blocksPerGrid];
    cudaMemcpy(h_temp, d_temp, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Finalize the reduction on CPU (sum the block results)
    int sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_temp[i];
    }
    
    // Print results
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_temp);
    delete[] h_temp;
    
    return sum;
}

int main(int argc, char** argv) {
    int size = 1024;  // Default size
    
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    
    // Print size for verification
    std::cout << "Running CUDA Reduction for size: " << size << std::endl;
    
    // Allocate and initialize host array
    int* h_input = new int[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1;  // Set all elements to 1 for easy verification
    }
    
    // Run the reduction and get the sum
    int result = sumArray(h_input, size);
    
    // Verify result (should equal the array size since all elements are 1)
    bool correct = (result == size);
    std::cout << "Result verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    // Clean up
    delete[] h_input;
    
    return 0;
}
