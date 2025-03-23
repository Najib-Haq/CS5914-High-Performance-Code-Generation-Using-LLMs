
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size)
{
    /*
    Optimization Strategy:
    1. **Grid-Stride Loop with Manual Unrolling**: Each thread processes multiple elements in the input array using a grid-stride loop with manual unrolling by a factor of 4. This reduces loop overhead and increases instruction-level parallelism, allowing threads to handle larger portions of the data efficiently.
    
    2. **Memory Coalescing**: Consecutive threads access consecutive memory locations, ensuring coalesced global memory accesses. This maximizes memory bandwidth utilization and reduces memory latency.
    
    3. **Intra-Warp Reduction Using Warp Shuffle Instructions (`__shfl_down_sync`)**: Utilizes warp-level primitives to perform reductions within a warp without relying on shared memory. This minimizes latency and avoids shared memory bank conflicts, leading to faster intra-warp reductions.
    
    4. **Per-Warp Partial Sums Stored in Shared Memory with Padding**: Each warp writes its partial sum to shared memory. Padding can be added if necessary to prevent shared memory bank conflicts, although with modern GPUs and careful access patterns, this is often minimized.
    
    5. **Final Block Reduction Using Warp Shuffle Instructions**: The first warp reads the partial sums from shared memory and performs a final reduction using warp shuffle instructions. This eliminates the need for additional synchronization and leverages the low latency of warp-level operations.
    
    6. **Minimized Synchronization and Divergent Execution**: By structuring the kernel to minimize the number of `__syncthreads()` calls and ensuring that all threads within a warp execute similar instructions, the kernel reduces synchronization overhead and avoids divergent execution paths, enhancing overall performance.
    */

    extern __shared__ int shared_data[];

    // Calculate thread and block indices
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gridSize = blockSize * gridDim.x;

    // Initialize local sum
    int sum = 0;

    // Grid-stride loop with manual unrolling by a factor of 4
    for(unsigned int i = tid; i < size; i += gridSize * 4)
    {
        if(i < size)
            sum += input[i];
        if(i + gridSize < size)
            sum += input[i + gridSize];
        if(i + 2 * gridSize < size)
            sum += input[i + 2 * gridSize];
        if(i + 3 * gridSize < size)
            sum += input[i + 3 * gridSize];
    }

    // Intra-warp reduction using warp shuffle
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Write the per-warp partial sum to shared memory
    if((tid % warpSize) == 0)
    {
        shared_data[tid / warpSize] = sum;
    }

    // Ensure all partial sums are written to shared memory
    __syncthreads();

    // Only the first warp performs the final reduction
    if(tid < (blockSize / warpSize))
    {
        // Load the partial sums from shared memory
        sum = shared_data[tid];

        // Final reduction using warp shuffle
        for(int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Write the final block sum to the output array
        if(tid == 0)
        {
            output[blockIdx.x] = sum;
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
