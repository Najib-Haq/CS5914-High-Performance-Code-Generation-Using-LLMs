__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Maximize parallel reduction efficiency through hybrid approach
    // 2. Use cooperative loading to reduce global memory accesses
    // 3. Leverage warp shuffle instructions for low-overhead reductions
    // 4. Minimize synchronization and branch divergence
    // 5. Handle non-power-of-2 sizes with predicated loading
    // 6. Use sequential shared memory addressing to reduce bank conflicts
    
    // Declare shared memory with careful layout to minimize bank conflicts
    extern __shared__ int sharedData[];
    
    // Thread and global index calculations
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Cooperative loading with boundary checks and two-element per thread load
    int localSum = 0;
    
    // First cooperative load with boundary check
    if (globalIdx < size) {
        localSum = input[globalIdx];
    }
    
    // Second cooperative load, avoiding redundant size check
    if (globalIdx + blockDim.x < size) {
        localSum += input[globalIdx + blockDim.x];
    }
    
    // Warp-level reduction using shuffle instructions
    // Aggressive unrolling for instruction-level parallelism
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    
    // Store warp reduction results in shared memory
    // Use sequential addressing to minimize bank conflicts
    if (tid % warpSize == 0) {
        sharedData[tid / warpSize] = localSum;
    }
    
    // Block-level synchronization
    __syncthreads();
    
    // Final block reduction using first warp
    if (tid < warpSize) {
        // Load warp reduction results, handling potential empty warps
        localSum = (tid < (blockDim.x * 2 + warpSize - 1) / warpSize) ? 
                   sharedData[tid] : 0;
        
        // Final warp-level shuffle reduction
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            localSum += __shfl_down_sync(0xffffffff, localSum, offset);
        }
        
        // First thread writes block result
        if (tid == 0) {
            output[blockIdx.x] = localSum;
        }
    }
}