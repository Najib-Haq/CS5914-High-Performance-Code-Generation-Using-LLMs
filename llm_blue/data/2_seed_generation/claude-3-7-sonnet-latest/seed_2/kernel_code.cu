__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use sequential addressing pattern to avoid bank conflicts in shared memory
    // 2. Process multiple elements per thread during loading phase using grid-stride loops
    // 3. Use thread coarsening - each thread handles multiple elements initially
    // 4. Reduce shared memory transactions with first-level reduction during loading
    // 5. Employ warp shuffle operations for the final warp to eliminate shared memory access and sync barriers
    // 6. Minimize divergent execution with carefully structured conditionals
    // 7. Use compile-time loop unrolling for the last warp reduction
    // 8. Eliminate unnecessary __syncthreads() calls in the final warp
    
    extern __shared__ int sdata[];
    
    // Thread identification
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    // Initialize thread-local sum
    int thread_sum = 0;
    
    // Grid-stride loop to handle arrays of any size
    // Each thread processes multiple elements sequentially, improving coalesced memory access
    for (unsigned int i = globalIdx; i < size; i += gridSize) {
        thread_sum += input[i];
    }
    
    // Store thread sum to shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree-based reduction in shared memory with sequential addressing
    // Only active threads perform work, reducing warp divergence
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using warp-level primitives
    // Eliminates shared memory accesses and __syncthreads() for higher performance
    if (tid < 32) {
        // Use volatile pointer for pre-Volta GPUs that don't have independent thread scheduling
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        
        // Use warp shuffle operations for the final reduction steps
        int value = sdata[tid];
        
        // Unroll the last 5 iterations (warp size = 32, so we need 5 steps: 16,8,4,2,1)
        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value += __shfl_down_sync(0xffffffff, value, 2);
        value += __shfl_down_sync(0xffffffff, value, 1);
        
        // First thread in block writes the result to global memory
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}