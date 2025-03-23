__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use 1024 threads per block to maximize memory bandwidth utilization and strike a balance between memory access and computational throughput.
    // 2. Implement a bank-conflict-free memory access pattern using a strided addressing pattern with stride 32 (number of threads per warp).
    // 3. Utilize warp-level primitives like __shfl_down_sync() for efficient warp-level reductions.
    // 4. Employ a combination of warp-level and block-level reductions to minimize the number of threads participating in the final reduction phase.
    // 5. Apply an early exit strategy to minimize unnecessary work when the input size is less than the number of threads per block.
    // 6. Unroll the reduction loop to reduce loop overhead and increase instruction-level parallelism.
    // This strategy should be effective because it reduces memory traffic, optimizes computations, and maximizes utilization of GPU resources.

    extern __shared__ int sharedData[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Early exit strategy: If the size of the input array is less than the number of threads per block,
    // only the first 'size' threads participate in the reduction.
    if (idx >= size) return;

    // Load data from global memory using a coalesced access pattern
    int sum = 0;
    while (idx < size) {
        sum += input[idx];
        idx += stride;
    }

    // Store loaded data in shared memory using a bank-conflict-free strided addressing pattern
    sharedData[threadIdx.x] = sum;
    __syncthreads();

    // Warp-level reduction using __shfl_down_sync()
    for (int offset = 512; offset > 0; offset /= 2) {
        int neighborVal = __shfl_down_sync(0xffffffff, sharedData[threadIdx.x], offset);
        if (threadIdx.x < offset) {
            sharedData[threadIdx.x] += neighborVal;
        }
    }

    // Reduction across warps in the block
    if (threadIdx.x < 32) {
        int warpId = threadIdx.x;
        sharedData[warpId] += sharedData[warpId + 32];
        sharedData[warpId] += sharedData[warpId + 64];
        sharedData[warpId] += sharedData[warpId + 96];
        sharedData[warpId] += sharedData[warpId + 128];
        sharedData[warpId] += sharedData[warpId + 160];
        sharedData[warpId] += sharedData[warpId + 192];
        sharedData[warpId] += sharedData[warpId + 224];
        sharedData[warpId] += sharedData[warpId + 256];
        sharedData[warpId] += sharedData[warpId + 288];
        sharedData[warpId] += sharedData[warpId + 320];
        sharedData[warpId] += sharedData[warpId + 352];
        sharedData[warpId] += sharedData[warpId + 384];
        sharedData[warpId] += sharedData[warpId + 416];
        sharedData[warpId] += sharedData[warpId + 448];
        sharedData[warpId] += sharedData[warpId + 480];
        sharedData[warpId] += sharedData[warpId + 512];
        sharedData[warpId] += sharedData[warpId + 544];
        sharedData[warpId] += sharedData[warpId + 576];
        sharedData[warpId] += sharedData[warpId + 608];
        sharedData[warpId] += sharedData[warpId + 640];
        sharedData[warpId] += sharedData[warpId + 672];
        sharedData[warpId] += sharedData[warpId + 704];
        sharedData[warpId] += sharedData[warpId + 736];
        sharedData[warpId] += sharedData[warpId + 768];
        sharedData[warpId] += sharedData[warpId + 800];
        sharedData[warpId] += sharedData[warpId + 832];
        sharedData[warpId] += sharedData[warpId + 864];
        sharedData[warpId] += sharedData[warpId + 896];
        sharedData[warpId] += sharedData[warpId + 928];
        sharedData[warpId] += sharedData[warpId + 960];
        sharedData[warpId] += sharedData[warpId + 992];
    }
    __syncthreads();

    // Store the block's result in the output array
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}