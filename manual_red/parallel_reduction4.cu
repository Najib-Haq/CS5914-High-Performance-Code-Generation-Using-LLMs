#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

// Warp-Level Reduction Using Template Metaprogramming for Full Unrolling
template <int Offset>
__inline__ __device__ void warpReduceSumUnrolled(int &sum) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, Offset);
}

// Fully Unrolled Block-Level Reduction Using Template Metaprogramming
template <int BlockSize>
__inline__ __device__ void blockReduceSum(int* shared_data, int tid) {
    // these if‐constexprs get resolved at compile time into straight code
    if constexpr (BlockSize >= 1024) {
        if (tid < 512) shared_data[tid] += shared_data[tid + 512];
        __syncthreads();
    }
    if constexpr (BlockSize >= 512) {
        if (tid < 256) shared_data[tid] += shared_data[tid + 256];
        __syncthreads();
    }
    if constexpr (BlockSize >= 256) {
        if (tid < 128) shared_data[tid] += shared_data[tid + 128];
        __syncthreads();
    }
    if constexpr (BlockSize >= 128) {
        if (tid < 64) shared_data[tid] += shared_data[tid + 64];
        __syncthreads();
    }
    // final warp
    if (tid < 32) {
        // fully unroll the last 32→1 reduction in‑warp
        warpReduceSumUnrolled<32>(shared_data[tid]);
        warpReduceSumUnrolled<16>(shared_data[tid]);
        warpReduceSumUnrolled<8>(shared_data[tid]);
        warpReduceSumUnrolled<4>(shared_data[tid]);
        warpReduceSumUnrolled<2>(shared_data[tid]);
        warpReduceSumUnrolled<1>(shared_data[tid]);
    }
}


__global__ void sumReduction(int* input, int* output, int n) {
    extern __shared__ int sdata[];  // shared mem, size = blockDim.x

    const unsigned int B    = blockDim.x;
    const unsigned int tid  = threadIdx.x;
    const unsigned int base = blockIdx.x * (2 * B);

    // each thread sums two input elements into sdata[tid]
    unsigned int idx1 = base + tid;           // first element
    unsigned int idx2 = base + tid + B;       // second element
    int val = 0;
    if (idx1 < (unsigned)n) val += input[idx1];
    if (idx2 < (unsigned)n) val += input[idx2];
    sdata[tid] = val;
    __syncthreads();

    // 2) Fully unrolled block reduction
    blockReduceSum<THREADS_PER_BLOCK>(sdata, tid);

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


// Host function to perform reduction
int sumArray(int* h_input, int size) {
    int *d_input, *d_temp;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, size * sizeof(int));
    
    // The size of d_temp is based on the number of blocks we'll launch
    int threadsPerBlock = THREADS_PER_BLOCK;
    int elemsPerBlock   = threadsPerBlock * 2;  // coarsened load - as each thread sums two elements during loading in shared memory
    int blocksPerGrid   = (size + elemsPerBlock - 1) / elemsPerBlock;
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