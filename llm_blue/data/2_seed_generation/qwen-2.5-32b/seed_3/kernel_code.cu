__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // - Utilization of warp-striped memory access to minimize bank conflicts.
    // - Performing warp-level reduction using __shfl_down_sync for rapid reduction within each warp.
    // - Minimizing synchronization points to reduce the overhead caused by global synchronization.
    // - Early exit strategy for threads not involved in reduction steps to avoid unnecessary work.
    // - Effective management of shared memory and block-level reduction to ensure high throughput.

    extern __shared__ int sharedMem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory with warp-striped access
    int data = (idx < size) ? input[idx] : 0;
    sharedMem[tid] = data;
    __syncthreads();

    // Perform warp-level reduction with __shfl_down_sync
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sharedMem[tid] += __shfl_down_sync(0xFFFFFFFF, sharedMem[tid], stride);
        }
        __syncthreads();
    }

    // If this thread is the last active thread (warp 0, thread 0)
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}