
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use a sufficient number of threads per block to fully utilize the GPU's processing capabilities.
    // 2. Implement a bank-conflict-free memory access pattern in shared memory using a sequential addressing pattern.
    // 3. Utilize warp-level primitives like __shfl_down_sync() for efficient warp-level reductions.
    // 4. Apply an early exit strategy to minimize unnecessary work when the input size is less than the number of threads per block.
    // 5. Employ a combination of warp-level and block-level reductions to minimize the number of threads participating in the final reduction phase.
    // This strategy should be effective because it reduces memory traffic, optimizes computations, and maximizes utilization of GPU resources.

    extern __shared__ int sharedData[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Early exit strategy: If the size of the input array is less than the number of threads per block,
    // only the first 'size' threads participate in the reduction.
    if (idx >= size) return;

    // Load data from global memory
    int sum = 0;
    while (idx < size) {
        sum += input[idx];
        idx += stride;
    }

    // Store loaded data in shared memory using a sequential addressing pattern to minimize bank conflicts
    sharedData[threadIdx.x] = sum;
    __syncthreads();

    // Warp-level reduction using __shfl_down_sync()
    for (int offset = 16; offset > 0; offset /= 2) {
        int neighborVal = __shfl_down_sync(0xffffffff, sharedData[threadIdx.x], offset);
        if (threadIdx.x % 32 < offset) {
            sharedData[threadIdx.x] += neighborVal;
        }
    }

    // Reduction across warps in the block
    if (threadIdx.x % 32 == 0) {
        int warpId = threadIdx.x / 32;
        for (int i = 1; i < 8; i++) { // 8 warps per block (256 threads / 32 threads per warp)
            if (warpId < i) {
                sharedData[warpId * 32] += sharedData[i * 32];
            }
        }
    }

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
