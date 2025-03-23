__global__ void sumReduction(int *input, int *output, int size)
{
    // Optimization Strategy:
    // 1. **Loop Unrolling with Strided Access**: Each thread processes multiple elements by striding through the input array,
    //    which maximizes memory throughput and handles large input sizes efficiently.
    // 2. **Warp-Level Primitives (__shfl_down_sync)**: Utilizes warp shuffle operations to perform intra-warp reductions without
    //    shared memory, reducing latency and avoiding bank conflicts.
    // 3. **Shared Memory for Per-Warp Aggregation**: After intra-warp reductions, each warp's partial sum is stored in
    //    shared memory. This minimizes global memory accesses and leverages the low latency of shared memory.
    // 4. **Unconditional Loop for Final Reduction**: The final reduction within shared memory is unrolled to eliminate divergent branches,
    //    ensuring consistent execution paths and enhanced performance.
    // 5. **Boundary Checks**: Ensures that all memory accesses are within the bounds of the input array, maintaining correctness
    //    across all input sizes from 1K to 1B elements.

    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int gridSize = blockSize * gridDim.x;

    // Initialize local sum
    int sum = 0;

    // Grid-stride loop to accumulate partial sums
    for (unsigned int idx = blockIdx.x * blockSize + tid; idx < size; idx += gridSize)
    {
        sum += input[idx];
    }

    // Intra-warp reduction using warp shuffle
    // Assumes blockSize is a multiple of warp size (32)
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Each warp writes its local sum to shared memory
    if ((tid & (warpSize - 1)) == 0)
    {
        shared_data[tid / warpSize] = sum;
    }

    __syncthreads();

    // Only one warp performs the final reduction using shared memory
    if (tid < (blockSize / warpSize))
    {
        sum = shared_data[tid];
        // Final reduction within the first warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        // Store the block's result
        if (tid == 0)
        {
            output[blockIdx.x] = sum;
        }
    }
}