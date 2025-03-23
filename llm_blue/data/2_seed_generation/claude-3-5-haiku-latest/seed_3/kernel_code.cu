__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Hybrid reduction approach combining warp shuffle and shared memory
    // 2. Maximize parallel reduction efficiency
    // 3. Minimize global memory accesses and synchronization overhead
    // 4. Handle non-power-of-2 sizes with predicated loading
    // 5. Reduce bank conflicts through careful shared memory layout
    // 6. Leverage instruction-level parallelism with aggressive unrolling
    
    // Declare shared memory with sequential addressing to reduce bank conflicts
    extern __shared__ int sharedData[];
    
    // Thread and index calculations
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Cooperative loading with boundary checks and sequential memory access
    int value = 0;
    if (globalIdx < size) {
        value = input[globalIdx];
        // Load two elements per thread to increase computational efficiency
        if (globalIdx + blockDim.x < size) {
            value += input[globalIdx + blockDim.x];
        }
    }
    
    // Initial warp-level reduction using shuffle instructions
    // Unrolled for maximum performance
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // Store warp reduction results in shared memory
    if (tid % warpSize == 0) {
        sharedData[tid / warpSize] = value;
    }
    
    // Block-level synchronization
    __syncthreads();
    
    // Final block reduction using first warp
    if (tid < warpSize) {
        // Load warp reduction results, handling potential empty warps
        value = (tid < (blockDim.x * 2 + warpSize - 1) / warpSize) ? 
                sharedData[tid] : 0;
        
        // Final warp-level shuffle reduction with aggressive unrolling
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        // First thread writes block result
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}