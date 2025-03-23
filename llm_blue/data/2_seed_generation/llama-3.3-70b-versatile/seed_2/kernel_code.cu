__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use a sufficient number of threads per block to fully utilize the GPU's processing capabilities.
    // 2. Implement a bank-conflict-free memory access pattern in shared memory using a sequential addressing pattern.
    // 3. Utilize warp-level primitives like __shfl_down_sync() for efficient warp-level reductions.
    // 4. Apply an early exit strategy to minimize unnecessary work when the input size is less than the number of threads per block.
    // 5. Employ a combination of warp-level and block-level reductions to minimize the number of threads participating in the final reduction phase.
    // This strategy should be effective because it reduces memory traffic, optimizes computations, and maximizes utilization of GPU resources.

    extern __shared__ int sharedData[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Early exit strategy: If the size of the input array is less than the number of threads per block,
    // only the first 'size' threads participate in the reduction.
    if (idx >= size) return;

    // Load data from global memory
    int sum = 0;
    while (idx < size) {
        sum += input[idx];
        idx += stride;
    }

    // Store loaded data in shared memory using a sequential addressing pattern to minimize bank conflicts
    sharedData[threadIdx.x] = sum;
    __syncthreads();

    // Warp-level reduction using __shfl_down_sync()
    for (int offset = 16; offset > 0; offset /= 2) {
        int neighborVal = __shfl_down_sync(0xffffffff, sharedData[threadIdx.x], offset);
        if (threadIdx.x % 32 < offset) {
            sharedData[threadIdx.x] += neighborVal;
        }
    }

    // Reduction across warps in the block
    if (threadIdx.x % 32 == 0) {
        int warpId = threadIdx.x / 32;
        for (int i = 1; i < 8; i++) { // 8 warps per block (256 threads / 32 threads per warp)
            if (warpId < i) {
                sharedData[warpId * 32] += sharedData[i * 32];
            }
        }
    }

    // Store the block's result in the output array
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}