__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Use shared memory to reduce global memory accesses
    // 2. Implement a tree-based parallel reduction algorithm
    // 3. Handle non-power-of-2 array sizes with boundary checks
    // 4. Use warp-level optimizations and minimize synchronization
    
    // Declare shared memory dynamically
    extern __shared__ int sharedData[];
    
    // Global thread and block indices
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory, with boundary protection
    sharedData[tid] = (globalIdx < size) ? input[globalIdx] : 0;
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    // Use log2(blockDim.x) iterations for complete reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        
        // Synchronize after each reduction step
        __syncthreads();
    }
    
    // The first thread in the block writes the block's reduction result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}