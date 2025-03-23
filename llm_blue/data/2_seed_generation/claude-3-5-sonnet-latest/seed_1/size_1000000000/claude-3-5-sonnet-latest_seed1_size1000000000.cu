
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
/*
 * Optimization strategy:
 * 1. Uses warp-level primitives (__shfl_down_sync) to eliminate shared memory and sync in final warp
 * 2. Processes 8 elements per thread during load to reduce total thread count and memory transactions
 * 3. Uses sequential addressing to prevent shared memory bank conflicts
 * 4. Employs template-style loop unrolling for the reduction phase
 * 5. Minimizes divergent execution with predicated operations where possible
 * 6. Uses grid-stride loop for handling arbitrary input sizes efficiently
 * 7. Maintains coalesced memory access patterns during initial loading phase
 */
__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    
    // Thread and block index
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 8) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 8 * gridDim.x;
    
    // Initialize thread-local sum
    int thread_sum = 0;

    // Grid-stride loop, each thread processes 8 elements at a time
    while (idx < size) {
        int temp_sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int current = idx + i * blockDim.x;
            if (current < size) {
                temp_sum += input[current];
            }
        }
        thread_sum += temp_sum;
        idx += gridSize;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    #pragma unroll
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle
    if (tid < 32) {
        int value = sdata[tid];
        value += tid + 32 < blockDim.x ? sdata[tid + 32] : 0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
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
