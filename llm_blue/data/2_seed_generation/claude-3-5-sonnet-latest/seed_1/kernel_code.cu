/*
 * Optimization strategy:
 * 1. Uses warp-level primitives (__shfl_down_sync) to eliminate shared memory and sync in final warp
 * 2. Processes 8 elements per thread during load to reduce total thread count and memory transactions
 * 3. Uses sequential addressing to prevent shared memory bank conflicts
 * 4. Employs template-style loop unrolling for the reduction phase
 * 5. Minimizes divergent execution with predicated operations where possible
 * 6. Uses grid-stride loop for handling arbitrary input sizes efficiently
 * 7. Maintains coalesced memory access patterns during initial loading phase
 */
__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sdata[];
    
    // Thread and block index
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 8) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 8 * gridDim.x;
    
    // Initialize thread-local sum
    int thread_sum = 0;

    // Grid-stride loop, each thread processes 8 elements at a time
    while (idx < size) {
        int temp_sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int current = idx + i * blockDim.x;
            if (current < size) {
                temp_sum += input[current];
            }
        }
        thread_sum += temp_sum;
        idx += gridSize;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    #pragma unroll
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle
    if (tid < 32) {
        int value = sdata[tid];
        value += tid + 32 < blockDim.x ? sdata[tid + 32] : 0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        
        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}