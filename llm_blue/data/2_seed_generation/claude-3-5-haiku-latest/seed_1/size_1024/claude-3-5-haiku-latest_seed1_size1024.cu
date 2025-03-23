
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Use warp-level shuffle instructions for faster intra-warp reduction
    // 2. Minimize global memory access and synchronization
    // 3. Handle non-power-of-2 sizes with predicated load
    // 4. Unroll reduction stages for better instruction-level parallelism
    // 5. Reduce shared memory bank conflicts with careful memory layout

    // Declare shared memory for block-level reduction
    extern __shared__ int sharedData[];
    
    // Global and local thread indices
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Predicated load with boundary check
    int value = (globalIdx < size) ? input[globalIdx] : 0;
    
    // Warp-level reduction using shuffle instructions
    // Reduces synchronization overhead and improves performance
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // Store warp-level reduction result in shared memory
    if (tid % warpSize == 0) {
        sharedData[tid / warpSize] = value;
    }
    
    // Block-level synchronization
    __syncthreads();
    
    // Final block reduction
    if (tid < warpSize) {
        value = (tid < (blockDim.x + warpSize - 1) / warpSize) ? 
                sharedData[tid] : 0;
        
        // Warp-level shuffle reduction for remaining warps
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        // First thread writes block result
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
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
