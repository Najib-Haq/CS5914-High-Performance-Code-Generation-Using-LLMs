#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sumReduction(int* input, int* output, int n) {
    const unsigned int B    = blockDim.x;  // Number of threads per block
    const unsigned int base = blockIdx.x * B; // Base index for this block
    const unsigned int tid  = threadIdx.x; // Thread index within the block
    const unsigned int idx  = base + tid; // Global index for this thread

    // If this thread’s element is past the end, do nothing
    if (idx >= (unsigned)n) return;

    for (unsigned int stride = 1; stride < B; stride <<= 1) {
        // only threads whose tid is multiple of 2*stride participate
        if ((tid % (2 * stride)) == 0) {
            unsigned int other = idx + stride;
            if (other < (unsigned)n) {
                input[idx] += input[other];
            }
        }
        __syncthreads();
    }

    // thread 0 of each block writes out the block’s sum
    if (tid == 0) {
        output[blockIdx.x] = input[base];
    }
}

// Host function to perform reduction
int sumArray(int* h_input, int size) {
    int *d_input, *d_temp;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(int));
    
    // The size of d_temp is based on the number of blocks we'll launch
    int threadsPerBlock = 1024;
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Wrong Usage: " << argv[0] << " <size>\n";
        return 1;
    }
    const int size = atoll(argv[1]); 
    if (size <= 0) {
        std::cerr << "Error: Invalid input size.\n";
        return 1;
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