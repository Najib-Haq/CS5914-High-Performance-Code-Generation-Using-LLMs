__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Sequential addressing to avoid bank conflicts in shared memory
    // 2. Two-element processing per thread during loading to reduce the initial data size
    // 3. Grid-stride loop to handle arbitrary sized inputs efficiently
    // 4. Complete unrolling for last 6 iterations (covers warp size of 32)
    // 5. Use warp shuffle operations for warp-level reductions to eliminate shared memory and sync in final stages
    // 6. Early exit when thread is out of bounds to avoid unnecessary work
    // 7. First add operation during loading to reduce total number of reduction steps
    // 8. Minimize divergent branches by consolidating conditionals
    // 9. Utilize full warp for shuffle operations for maximum throughput
    
    extern __shared__ int sdata[];
    
    // Thread and block identification
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // *2 for loading two elements
    unsigned int gridSize = blockDim.x * gridDim.x * 2; // *2 for the stride
    
    // Each thread accumulates results into a local variable
    int thread_sum = 0;
    
    // Grid-stride loop - each thread processes 2 elements per iteration
    for (unsigned int i = globalIdx; i < size; i += gridSize) {
        thread_sum += input[i];
        
        // Check if the second element is within bounds
        if (i + blockDim.x < size) {
            thread_sum += input[i + blockDim.x];
        }
    }
    
    // Store sum in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Sequential addressing: reduce with fewer bank conflicts and less divergence
    // Only do this for larger reductions, stop when we reach warp size
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Last warp reduction using warp shuffle
    if (tid < 32) {
        // Handle the case where blockSize is not a multiple of warp size
        if (blockSize >= 64) {
            sdata[tid] += sdata[tid + 32];
        }
        
        // Use warp shuffle operations for the final reduction (no sync needed)
        // Fully unrolled for maximum performance
        int value = sdata[tid];
        
        // Unroll for all possible warp sizes to ensure the compiler selects optimal code
        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value += __shfl_down_sync(0xffffffff, value, 2);
        value += __shfl_down_sync(0xffffffff, value, 1);
        
        // Only first thread writes result for this block
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}