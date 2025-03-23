__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization Strategy:
    // 1. Utilize shared memory to store partial sums for each block, reducing global memory accesses.
    // 2. Each thread loads multiple elements (if possible) to maximize memory bandwidth and handle large input sizes.
    // 3. Perform a tree-based reduction within shared memory to efficiently sum the elements.
    // 4. Use synchronization (__syncthreads()) to ensure all threads have loaded their data before starting the reduction.
    // 5. Unroll the last warp to minimize synchronization overhead and take advantage of warp-level parallelism.
    // These optimizations collectively enhance performance across a wide range of input sizes by maximizing memory throughput and minimizing synchronization costs.

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load elements into shared memory, handling boundary conditions
    int sum = 0;
    if (idx < size)
        sum += input[idx];
    if (idx + blockDim.x < size)
        sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Unroll the last warp
    if (tid < 32) {
        volatile int* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write the block's result to the output array
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}