
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
__global__ void sumReduction(int *input, int *output, int size) {
  // Optimization Strategy:
  // This kernel utilizes a hybrid approach combining warp-level reductions with a tree-based reduction in shared memory.  It prioritizes minimizing global memory accesses and maximizing warp-level parallelism.  
  // Bank conflict-free access is ensured in shared memory using modulo operations. Early exit conditions handle cases where the input size is smaller than the block size, preventing unnecessary computations.
  //  The choice of block size is a compromise between maximizing occupancy and minimizing the number of blocks for the final reduction.

  const int warpSize = 32;
  extern __shared__ int sdata[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Early exit if input size is smaller than block size.
  if (i >= size) {
    sdata[tid] = 0;
    return;
  }

  // Load data from global memory to shared memory with bank-conflict-free access.
  sdata[tid] = input[i + (tid % warpSize) * (blockDim.x / warpSize)];
  __syncthreads();


  //Warp-level reduction using __shfl_down_sync
  int sum = sdata[tid];
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  //Only the first thread of each warp writes to shared memory.
  if (tid % warpSize == 0) {
    sdata[tid / warpSize] = sum;
  }
  __syncthreads();

  //Tree-based reduction in shared memory.  Optimized for power-of-two block sizes for simplicity.
  int numWarps = blockDim.x / warpSize;
  for (int s = numWarps / 2; s > 0; s >>= 1) {
    if (tid / warpSize < s) {
      sdata[tid / warpSize] += sdata[tid / warpSize + s];
    }
    __syncthreads();
  }

  // Write the block sum to global memory
  if (tid / warpSize == 0) {
    output[blockIdx.x] = sdata[0];
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
