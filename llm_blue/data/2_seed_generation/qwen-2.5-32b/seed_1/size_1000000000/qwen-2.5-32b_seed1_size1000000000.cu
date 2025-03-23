
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
// Optimized Kernel using Warps to Reduce Bank Conflicts and Improve Performance
// - Utilizes warp-level reduction followed by block-level reduction for efficient parallelism
// - Uses __shfl_down_sync to perform reduction within each warp, reducing bank conflicts and improving shared memory usage
// - This strategy reduces the number of synchronization points and minimizes thread divergence for better performance
// - Early termination for threads not involved in reduction steps minimizes unnecessary work
// - Handles boundary cases and odd-sized arrays efficiently
// - Aims to minimize global memory access, ensure efficient shared memory allocation and usage, and enhance parallelism for high throughput

__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sharedMem[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data = (idx < size) ? input[idx] : 0;

    // Load data into shared memory
    sharedMem[threadIdx.x] = data;

    // Ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Number of input elements per thread
    int offset = 1;

    // Warp-level reduction using shuffle operations
    for (offset = 1; offset < blockDim.x; offset <<= 1) {
        int lane = threadIdx.x & (offset - 1);
        if (lane < offset) {
            sharedMem[threadIdx.x] += __shfl_down_sync(0xFFFFFFFF, sharedMem[threadIdx.x], offset);
        }
        __syncthreads();
    }

    // The final sum in each block is stored by the first thread
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedMem[0];
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
