/*
 * Optimization strategy:
 * 1. Uses sequential addressing to avoid bank conflicts in shared memory
 * 2. Unrolls the last warp to avoid unnecessary synchronization
 * 3. Uses template loop unrolling for reducing shared memory accesses
 * 4. Processes multiple elements per thread during load to reduce total threads needed
 * 5. Uses volatile shared memory for warp operations to ensure proper memory ordering
 */
__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    
    // Each thread loads and adds 4 elements to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 4 * gridDim.x;
    
    // Initialize shared memory with identity value
    sdata[tid] = 0;
    
    // Grid-stride loop to handle large arrays
    while (i < size) {
        int sum = 0;
        if (i < size) sum += input[i];
        if (i + blockDim.x < size) sum += input[i + blockDim.x];
        if (i + 2 * blockDim.x < size) sum += input[i + 2 * blockDim.x];
        if (i + 3 * blockDim.x < size) sum += input[i + 3 * blockDim.x];
        sdata[tid] += sum;
        i += gridSize;
    }
    __syncthreads();

    // Do reduction in shared memory
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    // Unroll last 6 iterations (warp is synchronized)
    if (tid < 32) {
        volatile int *smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}