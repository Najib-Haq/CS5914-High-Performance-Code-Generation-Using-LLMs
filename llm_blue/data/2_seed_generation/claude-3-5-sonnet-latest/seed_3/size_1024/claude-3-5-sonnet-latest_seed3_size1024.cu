
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// The kernel provided by the LLM
/*
 * Optimization strategy:
 * 1. Processes 32 elements per thread during load to maximize arithmetic intensity
 *    and minimize memory transactions (better for very large arrays)
 * 2. Uses three-stage reduction:
 *    a) Sequential accumulation during loading (32 elements)
 *    b) Warp-level reduction using shuffle operations
 *    c) Cross-warp reduction using minimal shared memory
 * 3. Employs vectorized loads (int4) where possible to improve memory throughput
 * 4. Uses grid-stride processing with careful boundary handling
 * 5. Minimizes shared memory usage to maximize occupancy
 * 6. Reduces synchronization points to absolute minimum
 * 7. Completely eliminates bank conflicts through sequential addressing
 * 8. Uses template metaprogramming-style unrolling for critical paths
 */
__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int wid = tid >> 5;
    const unsigned int gridSize = blockDim.x * gridDim.x * 32;
    unsigned int idx = blockIdx.x * (blockDim.x * 32) + tid;
    
    // Thread-local sum
    long long sum = 0; // Using long long to prevent overflow during accumulation
    
    // Grid-stride loop, each thread handles 32 elements
    while (idx < size) {
        // Vectorized loads where possible
        if (idx + 31 * blockDim.x < size && (idx % 4) == 0) {
            int4 in4;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                in4 = reinterpret_cast<int4*>(input)[idx/4 + i * blockDim.x/4];
                sum += in4.x + in4.y + in4.z + in4.w;
            }
        } else {
            // Regular loads for boundary cases
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                if (idx + i * blockDim.x < size) {
                    sum += input[idx + i * blockDim.x];
                }
            }
        }
        idx += gridSize;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write warp results to shared memory
    if (lane == 0) {
        sdata[wid] = static_cast<int>(sum);
    }
    __syncthreads();
    
    // Final reduction (only first warp)
    if (wid == 0) {
        sum = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0;
        
        #pragma unroll
        for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            output[blockIdx.x] = static_cast<int>(sum);
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
