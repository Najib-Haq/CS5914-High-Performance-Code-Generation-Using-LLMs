#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/* 
Adding 
- local shared memory to the kernel
- coarsened load to the kernel (each thread sums two elements during loading in shared memory)
- unrolling last warp - use __shfl_down_sync 
*/

// Warp‐Level Reduction using Unrolled Shuffle Instructions
__device__ void warpReduceSumUnrolled(volatile int* sdata, int tid) {
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 32);
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 16);
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 8);
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 4);
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 2);
    sdata[tid] += __shfl_down_sync(0xFFFFFFFF, sdata[tid], 1);  
}

__global__ void sumReduction(int* input, int* output, int n) {
    extern __shared__ int sdata[];            // shared mem, size = blockDim.x

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

    // tree reduction down to 32 threads
    for (unsigned int stride = B/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Final warp‐level unrolled shuffle reduction
    if (tid < 32) {
        warpReduceSumUnrolled(sdata, tid);       // fully reduce within the warp
    }

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
    int threadsPerBlock = 1024;
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
    const int size = atoll(argv[1]); //1342177280; //1342177280; // 1.34 billion elements (~5GB)
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