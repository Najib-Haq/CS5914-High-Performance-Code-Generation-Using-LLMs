__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // - The strategy employs a tree-based reduction algorithm using shared memory to minimize global memory access and to provide parallel reduction within a block.
    // - Each thread initially loads a segment of the input data into shared memory for fast access.
    // - The algorithm then reduces the data in multiple steps, halving the number of active threads with each step. This ensures that the reduction process is highly parallelized, making efficient use of GPU resources.
    // - External memory access only occurs at the very end to store the result to global memory, minimizing memory transaction overhead.

    extern __shared__ int sharedMem[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    int tempSum = (idx < size) ? input[idx] : 0;
    sharedMem[tid] = tempSum;
    __syncthreads();

    // Perform tree-based reduction
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 writes result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}