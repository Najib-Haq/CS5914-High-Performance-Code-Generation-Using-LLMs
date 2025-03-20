#include <cuda_runtime.h>
#include <iostream>

// Kernel for sum reduction
__global__ void sumReductionKernel(int *d_input, int *d_partial_sums) {
    // Define the number of elements processed by each block
    const int blockSize = 256;
    // Define the shared memory for partial sums
    __shared__ int sharedPartialSums[blockSize];

    // Calculate the global index
    int globalIndex = blockIdx.x * blockSize + threadIdx.x;

    // Initialize partial sum for this thread
    int partialSum = 0;

    // Iterate over input data to process
    for (int i = globalIndex; i < 1024; i += blockSize * gridDim.x) {
        // Accumulate the values for this thread
        partialSum += d_input[i];
    }

    // Store the partial sum in shared memory
    sharedPartialSums[threadIdx.x] = partialSum;

    // Synchronize the block
    __syncthreads();

    // Perform the reduction in shared memory
    int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    for (int stride = blockSize / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedPartialSums[threadIdx.x] += sharedPartialSums[threadIdx.x + stride];
        }
        // Synchronize the block to ensure all threads are at the same point
        __syncthreads();
    }

    // Store the result
    if (threadIdx.x == 0) {
        d_partial_sums[blockIdx.x] = sharedPartialSums[0];
    }
}

// Kernel for final sum reduction
__global__ void finalSumReductionKernel(int *d_partial_sums, int *d_final_sum) {
    // Define the number of blocks
    const int numBlocks = 4;

    // Define the shared memory for partial sums
    __shared__ int sharedPartialSums[256];

    // Calculate the global index
    int globalIndex = blockIdx.x * 256 + threadIdx.x;

    // Initialize partial sum for this thread
    int partialSum = 0;

    // Check if we're in a valid block
    if (blockIdx.x < numBlocks) {
        // Accumulate the values for this thread
        partialSum += d_partial_sums[blockIdx.x * 256 + threadIdx.x];
    }

    // Store the partial sum in shared memory
    sharedPartialSums[threadIdx.x] = partialSum;

    // Synchronize the block
    __syncthreads();

    // Perform the reduction in shared memory
    int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    for (int stride = 256 / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedPartialSums[threadIdx.x] += sharedPartialSums[threadIdx.x + stride];
        }
        // Synchronize the block to ensure all threads are at the same point
        __syncthreads();
    }

    // Store the result
    if (threadIdx.x == 0) {
        d_final_sum[0] += sharedPartialSums[0];
    }
}

int main() {
    // 1. Allocate and initialize an array of 1024 integers on the host.
    int h_input[1024];
    for (int i = 0; i < 1024; ++i) {
        h_input[i] = 1;
    }

    // 2. Allocate device memory for input, partial sums, and final sum
    int *d_input, *d_partial_sums, *d_final_sum;
    cudaMalloc((void **)&d_input, 1024 * sizeof(int));
    cudaMalloc((void **)&d_partial_sums, 4 * 256 * sizeof(int));
    cudaMalloc((void **)&d_final_sum, sizeof(int));

    // 3. Copy data from host to device.
    cudaMemcpy(d_input, h_input, 1024 * sizeof(int), cudaMemcpyHostToDevice);

    // 4. Create CUDA events to measure execution time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. Launch the sum reduction kernel.
    int blockSize = 256;
    int numBlocks = (1024 + blockSize - 1) / blockSize;
    cudaEventRecord(start);
    sumReductionKernel<<<4, 256>>>(d_input, d_partial_sums);
    finalSumReductionKernel<<<1, 256>>>(d_partial_sums, d_final_sum);
    cudaEventRecord(stop);

    // 6. Record the kernel execution time.
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 7. Copy the final sum back to the host.
    int h_final_sum;
    cudaMemcpy(&h_final_sum, d_final_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // 8. Print the sum and the kernel execution time.
    std::cout << "Final sum: " << h_final_sum << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds" << std::endl;

    // 9. Clean up device memory.
    cudaFree(d_input);
    cudaFree(d_partial_sums);
    cudaFree(d_final_sum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}