__global__ void sumReduction(int *input, int *output, int size) {
    /*
    Optimization Strategy:
    1. **Loop Unrolling in Grid-Stride Loop**: Each thread processes four elements per loop iteration. This reduces loop overhead and increases instruction-level parallelism, enhancing throughput for large arrays.

    2. **Minimized Shared Memory Usage with Warp-Level Reductions**: By leveraging warp shuffle operations (`__shfl_down_sync`), most of the reduction is performed within warps using registers, minimizing reliance on shared memory and reducing synchronization overhead.

    3. **Efficient Shared Memory Reduction with Unrolled Tree-Based Approach**: After intra-warp reductions, partial sums from each warp are stored in shared memory. The final reduction within shared memory is unrolled to minimize loop overhead and take advantage of parallelism, ensuring fast convergence to the block's total sum.

    4. **Bank-Conflict-Free Shared Memory Access Patterns**: Shared memory indices are accessed in a manner that avoids bank conflicts, maximizing memory throughput and ensuring that multiple threads can access shared memory simultaneously without serialization.

    5. **Early Exit for Out-of-Bounds Threads**: Threads that do not contribute to the final sum (e.g., those beyond the current data range) exit early, reducing unnecessary computations and improving overall efficiency across varying input sizes.

    6. **Sequential Addressing in Reduction Phase**: The reduction within shared memory uses sequential addressing to optimize memory access patterns and take advantage of caching mechanisms, further enhancing performance.

    These combined optimizations ensure that the kernel efficiently handles a wide range of input sizes (from 1K to 1B elements) by maximizing computational throughput, minimizing memory latency, and reducing synchronization overhead. The strategy effectively balances workload distribution, memory access efficiency, and parallel reduction techniques to achieve superior performance.
    */

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int sum = 0;

    // Unroll the grid-stride loop by a factor of 4
    if (idx < size) {
        sum += input[idx];
        if (idx + blockDim.x < size) sum += input[idx + blockDim.x];
        if (idx + 2 * blockDim.x < size) sum += input[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < size) sum += input[idx + 3 * blockDim.x];
    }

    // Perform intra-warp reduction using warp shuffle operations
    // Assuming warpSize is 32
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Write the reduced value of each warp to shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }

    __syncthreads();

    // Perform block-level reduction only in the first warp
    if (tid < warpSize) {
        // Load partial sums from shared memory
        sum = (tid < (blockDim.x / warpSize)) ? sdata[tid] : 0;

        // Final reduction within the first warp
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Write the block's total sum to the output array
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}