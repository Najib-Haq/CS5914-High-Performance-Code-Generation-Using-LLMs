__global__ void sumReduction(int *input, int *output, int size) {
    /*
    Optimization Strategy:
    1. **Increased Loop Unrolling Factor**: Each thread processes 8 elements per grid-stride loop iteration. This reduces loop overhead and increases arithmetic intensity, enhancing throughput for large input sizes (up to 1B elements).

    2. **Minimized Shared Memory Reduction Using Warp-Level Primitives**: After accumulating the local sum in registers, threads within a warp perform reductions using `__shfl_down_sync`, minimizing shared memory usage and synchronization overhead.

    3. **Efficient Shared Memory Utilization for Partial Sums**: Partial sums from each warp are stored in shared memory with a bank-conflict-free access pattern. This ensures fast storage and retrieval during the final reduction phase.

    4. **Final Reduction Handled by a Single Warp**: The first warp of each block performs the final reduction of partial sums stored in shared memory using warp shuffle operations, further minimizing synchronization and memory access overhead.

    5. **Streamlined Memory Access Patterns**: The grid-stride loop accesses global memory in a coalesced manner, while shared memory accesses are aligned to avoid bank conflicts, maximizing memory bandwidth utilization.

    6. **Early Boundary Checks**: Each thread ensures that it does not read beyond the input array boundaries, preventing memory access violations and avoiding unnecessary computations for out-of-bounds indices.

    By integrating these optimizations, the kernel achieves higher performance by maximizing utilization of computational resources, minimizing memory latency, and reducing synchronization and branching overhead. This ensures efficient scalability across a broad spectrum of input sizes, with particular improvements for very large arrays.
    */

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockId = blockIdx.x;
    unsigned int blockDim_x = blockDim.x;
    unsigned int global_tid = blockId * blockDim_x * 8 + tid;
    unsigned int stride = blockDim.x * 8 * gridDim.x;

    int sum = 0;

    // Unroll the grid-stride loop by a factor of 8
    for (unsigned int idx = global_tid; idx < size; idx += stride) {
        sum += input[idx];
        if (idx + blockDim.x * 1 < size) sum += input[idx + blockDim.x * 1];
        if (idx + blockDim.x * 2 < size) sum += input[idx + blockDim.x * 2];
        if (idx + blockDim.x * 3 < size) sum += input[idx + blockDim.x * 3];
        if (idx + blockDim.x * 4 < size) sum += input[idx + blockDim.x * 4];
        if (idx + blockDim.x * 5 < size) sum += input[idx + blockDim.x * 5];
        if (idx + blockDim.x * 6 < size) sum += input[idx + blockDim.x * 6];
        if (idx + blockDim.x * 7 < size) sum += input[idx + blockDim.x * 7];
    }

    // Perform reduction within the warp using shuffle operations
    // Assumes warpSize is 32
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    // Each warp's first thread writes its partial sum to shared memory
    int lane = tid % 32;
    if (lane == 0) {
        sdata[tid / 32] = sum;
    }

    __syncthreads();

    // Let the first warp handle the reduction of partial sums
    if (tid < 32) {
        // Load partial sums from shared memory
        sum = (tid < (blockDim_x / 32)) ? sdata[tid] : 0;

        // Final reduction using shuffle operations
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

        // Write the block's total sum to the output array
        if (tid == 0) {
            output[blockId] = sum;
        }
    }
}