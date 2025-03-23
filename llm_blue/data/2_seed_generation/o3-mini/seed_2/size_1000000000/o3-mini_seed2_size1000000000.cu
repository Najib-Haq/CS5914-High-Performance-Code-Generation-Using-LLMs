
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
    /*
    Optimization Strategy:
    1. **Grid-Stride Loop**: Each thread processes multiple elements by iterating over the input array with a stride equal to the total number of threads in the grid. This ensures efficient utilization of all threads, especially for very large arrays (up to 1B elements).
    
    2. **Per-Warp Reduction using Warp-Level Primitives**: Utilize `__shfl_down_sync` to perform reductions within each warp without requiring shared memory or synchronization. This leverages the fast warp shuffle instructions to minimize latency and avoid shared memory bank conflicts.
    
    3. **Shared Memory for Warp Aggregation**: After per-warp reductions, each warp's partial sum is written to shared memory. A single warp then performs a final reduction on these partial sums. This hierarchical reduction approach reduces the number of synchronization points and memory accesses.
    
    4. **Minimized Synchronization**: By confining most reductions to warp-level operations and limiting shared memory usage to a single step, the kernel minimizes the use of `__syncthreads()`, thereby reducing synchronization overhead.
    
    5. **Efficient Memory Access Patterns**: Accesses to global memory are coalesced through the grid-stride loop, ensuring maximum memory bandwidth utilization. Shared memory accesses are also optimized to be bank-conflict-free by aligning partial sums per warp.
    
    6. **Scalable for All Input Sizes**: The combined use of grid-stride loops and hierarchical reductions ensures that the kernel scales efficiently from small (1K) to very large (1B) input sizes without performance degradation.
    
    These optimizations collectively enhance performance by maximizing computational throughput, minimizing memory latency, and reducing synchronization overhead, making the kernel highly efficient across a wide range of input sizes.
    */

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    // Each thread computes a partial sum using a grid-stride loop
    int sum = 0;
    for (unsigned int idx = global_tid; idx < size; idx += gridSize) {
        sum += input[idx];
    }

    // Perform per-warp reduction using warp-level primitives
    // Assuming warpSize is 32
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // The first thread of each warp writes its partial sum to shared memory
    if ((threadIdx.x % warpSize) == 0) {
        sdata[threadIdx.x / warpSize] = sum;
    }

    __syncthreads();

    // Let the first warp handle the reduction of partial sums in shared memory
    if (threadIdx.x < (blockDim.x / warpSize)) {
        sum = (threadIdx.x < (blockDim.x / warpSize)) ? sdata[threadIdx.x] : 0;
        
        // Reduce the partial sums using warp-level primitives
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Write the block's total sum to the output array
        if (threadIdx.x == 0) {
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
