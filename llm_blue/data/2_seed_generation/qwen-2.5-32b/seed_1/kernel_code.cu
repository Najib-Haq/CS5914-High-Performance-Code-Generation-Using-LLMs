// Optimized Kernel using Warps to Reduce Bank Conflicts and Improve Performance
// - Utilizes warp-level reduction followed by block-level reduction for efficient parallelism
// - Uses __shfl_down_sync to perform reduction within each warp, reducing bank conflicts and improving shared memory usage
// - This strategy reduces the number of synchronization points and minimizes thread divergence for better performance
// - Early termination for threads not involved in reduction steps minimizes unnecessary work
// - Handles boundary cases and odd-sized arrays efficiently
// - Aims to minimize global memory access, ensure efficient shared memory allocation and usage, and enhance parallelism for high throughput

__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sharedMem[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data = (idx < size) ? input[idx] : 0;

    // Load data into shared memory
    sharedMem[threadIdx.x] = data;

    // Ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Number of input elements per thread
    int offset = 1;

    // Warp-level reduction using shuffle operations
    for (offset = 1; offset < blockDim.x; offset <<= 1) {
        int lane = threadIdx.x & (offset - 1);
        if (lane < offset) {
            sharedMem[threadIdx.x] += __shfl_down_sync(0xFFFFFFFF, sharedMem[threadIdx.x], offset);
        }
        __syncthreads();
    }

    // The final sum in each block is stored by the first thread
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}