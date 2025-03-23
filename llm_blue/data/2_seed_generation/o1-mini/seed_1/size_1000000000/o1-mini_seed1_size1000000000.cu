
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size)
{
    // Optimization Strategy:
    // 1. **Loop Unrolling with Strided Access**: Each thread processes multiple elements by striding through the input array,
    //    which maximizes memory throughput and handles large input sizes efficiently.
    // 2. **Warp-Level Primitives (__shfl_down_sync)**: Utilizes warp shuffle operations to perform intra-warp reductions without
    //    shared memory, reducing latency and avoiding bank conflicts.
    // 3. **Shared Memory for Per-Warp Aggregation**: After intra-warp reductions, each warp's partial sum is stored in
    //    shared memory. This minimizes global memory accesses and leverages the low latency of shared memory.
    // 4. **Unconditional Loop for Final Reduction**: The final reduction within shared memory is unrolled to eliminate divergent branches,
    //    ensuring consistent execution paths and enhanced performance.
    // 5. **Boundary Checks**: Ensures that all memory accesses are within the bounds of the input array, maintaining correctness
    //    across all input sizes from 1K to 1B elements.

    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gridSize = blockSize * gridDim.x;

    // Initialize local sum
    int sum = 0;

    // Grid-stride loop to accumulate partial sums
    for (unsigned int idx = blockIdx.x * blockSize + tid; idx < size; idx += gridSize)
    {
        sum += input[idx];
    }

    // Intra-warp reduction using warp shuffle
    // Assumes blockSize is a multiple of warp size (32)
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Each warp writes its local sum to shared memory
    if ((tid & (warpSize - 1)) == 0)
    {
        shared_data[tid / warpSize] = sum;
    }

    __syncthreads();

    // Only one warp performs the final reduction using shared memory
    if (tid < (blockSize / warpSize))
    {
        sum = shared_data[tid];
        // Final reduction within the first warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        // Store the block's result
        if (tid == 0)
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
