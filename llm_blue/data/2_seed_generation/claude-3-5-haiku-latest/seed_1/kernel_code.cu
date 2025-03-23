__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Use warp-level shuffle instructions for faster intra-warp reduction
    // 2. Minimize global memory access and synchronization
    // 3. Handle non-power-of-2 sizes with predicated load
    // 4. Unroll reduction stages for better instruction-level parallelism
    // 5. Reduce shared memory bank conflicts with careful memory layout

    // Declare shared memory for block-level reduction
    extern __shared__ int sharedData[];
    
    // Global and local thread indices
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Predicated load with boundary check
    int value = (globalIdx < size) ? input[globalIdx] : 0;
    
    // Warp-level reduction using shuffle instructions
    // Reduces synchronization overhead and improves performance
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // Store warp-level reduction result in shared memory
    if (tid % warpSize == 0) {
        sharedData[tid / warpSize] = value;
    }
    
    // Block-level synchronization
    __syncthreads();
    
    // Final block reduction
    if (tid < warpSize) {
        value = (tid < (blockDim.x + warpSize - 1) / warpSize) ? 
                sharedData[tid] : 0;
        
        // Warp-level shuffle reduction for remaining warps
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        // First thread writes block result
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}