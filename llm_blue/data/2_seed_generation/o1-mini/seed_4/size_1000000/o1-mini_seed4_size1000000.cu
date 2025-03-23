
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size)
{
    /*
    Optimization Strategy:
    1. **Memory Coalescing and Sequential Access**: Ensures that consecutive threads access consecutive memory locations, maximizing global memory bandwidth utilization and minimizing memory latency.

    2. **Loop Unrolling with Grid-Stride**: Implements a grid-stride loop with an unrolling factor of 8. This reduces loop overhead and allows each thread to process multiple elements per iteration, enhancing instruction-level parallelism and throughput for large input sizes.

    3. **Register Accumulation**: Accumulates partial sums in registers to minimize the reliance on shared memory during the initial accumulation phase, which reduces shared memory traffic and latency.

    4. **Warp-Level Reduction Using `__shfl_down_sync`**: Utilizes warp shuffle operations to perform efficient intra-warp reductions without shared memory, decreasing synchronization overhead and avoiding bank conflicts.

    5. **Shared Memory for Inter-Warp Reduction**: Stores partial sums from each warp into shared memory in a bank-conflict-free manner. This facilitates efficient inter-warp reduction with minimal synchronization.

    6. **Hierarchical Reduction Tree with Minimized Synchronization**: Performs a two-level reduction (intra-warp and inter-warp) to combine partial sums efficiently. Only necessary synchronization barriers (`__syncthreads()`) are used to ensure data consistency without introducing significant overhead.

    7. **Handling Arbitrary Input Sizes**: Employs boundary checks within the grid-stride loop to correctly handle input arrays of any size, preventing out-of-bounds memory accesses and ensuring correctness across all input sizes from 1K to 1B elements.

    8. **Optimal Shared Memory Usage**: Allocates shared memory based on the number of warps per block, ensuring efficient utilization without wastage. This is achieved by mapping one shared memory slot per warp.

    These combined optimizations aim to maximize memory throughput, reduce latency, and minimize synchronization overhead, thereby achieving superior performance, especially for very large input sizes.
    */

    extern __shared__ int shared_data[];

    // Calculate thread and block indices
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int warpSizeLocal = warpSize; // Typically 32
    unsigned int gridSize = blockSize * gridDim.x;

    // Initialize register for partial sum
    int sum = 0;

    // Grid-stride loop with unrolling factor of 8
    for (unsigned int idx = blockIdx.x * blockSize + tid; idx < size; idx += gridSize * 8)
    {
        if (idx + 0 * gridSize < size) sum += input[idx + 0 * gridSize];
        if (idx + 1 * gridSize < size) sum += input[idx + 1 * gridSize];
        if (idx + 2 * gridSize < size) sum += input[idx + 2 * gridSize];
        if (idx + 3 * gridSize < size) sum += input[idx + 3 * gridSize];
        if (idx + 4 * gridSize < size) sum += input[idx + 4 * gridSize];
        if (idx + 5 * gridSize < size) sum += input[idx + 5 * gridSize];
        if (idx + 6 * gridSize < size) sum += input[idx + 6 * gridSize];
        if (idx + 7 * gridSize < size) sum += input[idx + 7 * gridSize];
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warpSizeLocal / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Write the per-warp partial sum to shared memory
    if ((tid % warpSizeLocal) == 0)
    {
        // Calculate warp index within the block
        int warpIdx = tid / warpSizeLocal;
        shared_data[warpIdx] = sum;
    }

    // Ensure all partial sums are written to shared memory
    __syncthreads();

    // Inter-warp reduction in shared memory
    if (tid < (blockSize / warpSizeLocal))
    {
        // Load partial sums from shared memory
        int blockSum = shared_data[tid];

        // Perform reduction within the first warp
        for (int offset = warpSizeLocal / 2; offset > 0; offset /= 2)
        {
            blockSum += __shfl_down_sync(0xFFFFFFFF, blockSum, offset);
        }

        // Write the block's total sum to the output array
        if (tid == 0)
        {
            output[blockIdx.x] = blockSum;
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
