
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
/*
 * Optimization strategy:
 * 1. Uses 16 elements per thread during load to significantly reduce memory transactions
 *    and total thread count (better for very large arrays)
 * 2. Implements two-stage reduction: first parallel within warps using shuffle,
 *    then across warps using shared memory
 * 3. Completely eliminates shared memory bank conflicts using sequential addressing
 * 4. Minimizes divergent execution by removing most conditional statements
 * 5. Uses grid-stride loop with unrolled inner loop for handling large arrays
 * 6. Employs full warp-level primitives to eliminate synchronization in final stages
 * 7. Reduces register pressure by reusing variables where possible
 * 8. Maintains coalesced memory access patterns during initial loading
 */
__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid >> 5;  // Warp ID
    const unsigned int lane = tid & 31;  // Lane within warp
    const unsigned int gridSize = blockDim.x * gridDim.x * 16;
    unsigned int idx = blockIdx.x * (blockDim.x * 16) + tid;
    
    // Thread-local sum
    int sum = 0;
    
    // Grid-stride loop, each thread handles 16 elements
    while (idx < size) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int curr_idx = idx + i * blockDim.x;
            if (curr_idx < size) {
                sum += input[curr_idx];
            }
        }
        idx += gridSize;
    }
    
    // First reduction using warp shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write warp results to shared memory
    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x >> 5)) {  // Only first warp
        sum = (tid < (blockDim.x >> 5)) ? sdata[tid] : 0;
        
        #pragma unroll
        for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
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
