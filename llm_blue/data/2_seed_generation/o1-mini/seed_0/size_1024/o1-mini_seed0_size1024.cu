
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size)
{
    // Optimization Strategy:
    // 1. Utilize shared memory (extern __shared__) to store partial sums for faster access.
    // 2. Each thread processes two elements to reduce the number of required iterations.
    // 3. Apply tree-based reduction within shared memory to efficiently sum elements.
    // 4. Unroll the last warp to minimize synchronization overhead and maximize performance.
    // 5. Handle boundaries by checking against 'size' to ensure correctness for all input sizes.

    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load elements into shared memory, processing two elements per thread
    int sum = 0;
    if (idx < size)
        sum += input[idx];
    if (idx + blockDim.x < size)
        sum += input[idx + blockDim.x];
    
    shared_data[tid] = sum;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
            shared_data[tid] += shared_data[tid + stride];
        __syncthreads();
    }

    // Unroll the last warp for maximum performance
    if (tid < 32)
    {
        volatile int* sm = shared_data;
        sm[tid] += sm[tid + 32];
        sm[tid] += sm[tid + 16];
        sm[tid] += sm[tid + 8];
        sm[tid] += sm[tid + 4];
        sm[tid] += sm[tid + 2];
        sm[tid] += sm[tid + 1];
    }

    // Write the result of this block to the output array
    if (tid == 0)
        output[blockIdx.x] = shared_data[0];
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
