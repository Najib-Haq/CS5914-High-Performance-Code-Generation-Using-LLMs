#include <cuda_runtime.h>
#include <iostream>

// Sum reduction kernel using shared memory
__global__ void sumReductionKernel(int *d_input, int *d_output) {
    // Shared memory allocation
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = d_input[idx];
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_output[0] = sdata[0];
    }
}

int main() {
    const int ARRAY_SIZE = 1024;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // 1. Allocate and initialize an array of 1024 integers on the host.
    int h_input[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        h_input[i] = 1;
    }
    int h_sum = 0;

    // 2. Allocate device memory.
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, ARRAY_BYTES);
    cudaMalloc((void**)&d_output, sizeof(int));

    // 3. Copy data from host to device.
    cudaMemcpy(d_input, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // 4. Create CUDA events to measure execution time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. Launch the sum reduction kernel.
    int threadsPerBlock = 1024;
    int blocksPerGrid = 1;

    cudaEventRecord(start);
    sumReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaEventRecord(stop);

    // 6. Record the kernel execution time.
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 7. Copy the final sum back to the host.
    cudaMemcpy(&h_sum, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // 8. Print the sum and the kernel execution time.
    std::cout << "Sum: " << h_sum << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 9. Clean up device memory.
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}