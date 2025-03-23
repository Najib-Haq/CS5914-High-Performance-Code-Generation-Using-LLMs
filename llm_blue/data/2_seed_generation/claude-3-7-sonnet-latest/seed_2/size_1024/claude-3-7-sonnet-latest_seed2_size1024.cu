
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use sequential addressing pattern to avoid bank conflicts in shared memory
    // 2. Process multiple elements per thread during loading phase using grid-stride loops
    // 3. Use thread coarsening - each thread handles multiple elements initially
    // 4. Reduce shared memory transactions with first-level reduction during loading
    // 5. Employ warp shuffle operations for the final warp to eliminate shared memory access and sync barriers
    // 6. Minimize divergent execution with carefully structured conditionals
    // 7. Use compile-time loop unrolling for the last warp reduction
    // 8. Eliminate unnecessary __syncthreads() calls in the final warp
    
    extern __shared__ int sdata[];
    
    // Thread identification
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    // Initialize thread-local sum
    int thread_sum = 0;
    
    // Grid-stride loop to handle arrays of any size
    // Each thread processes multiple elements sequentially, improving coalesced memory access
    for (unsigned int i = globalIdx; i < size; i += gridSize) {
        thread_sum += input[i];
    }
    
    // Store thread sum to shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree-based reduction in shared memory with sequential addressing
    // Only active threads perform work, reducing warp divergence
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using warp-level primitives
    // Eliminates shared memory accesses and __syncthreads() for higher performance
    if (tid < 32) {
        // Use volatile pointer for pre-Volta GPUs that don't have independent thread scheduling
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        
        // Use warp shuffle operations for the final reduction steps
        int value = sdata[tid];
        
        // Unroll the last 5 iterations (warp size = 32, so we need 5 steps: 16,8,4,2,1)
        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value += __shfl_down_sync(0xffffffff, value, 2);
        value += __shfl_down_sync(0xffffffff, value, 1);
        
        // First thread in block writes the result to global memory
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
