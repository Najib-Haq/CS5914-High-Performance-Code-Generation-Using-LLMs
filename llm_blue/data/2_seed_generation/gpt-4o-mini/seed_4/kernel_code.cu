__global__ void sumReduction(int *input, int *output, int size) {
    // Define shared memory for storing input data for this block
    extern __shared__ int sharedData[];

    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // Load input elements into shared memory, handling boundaries
    sharedData[localTid] = (tid < size) ? input[tid] : 0;

    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    // Early exit for a single thread case
    if (blockDim.x == 1) {
        if (localTid == 0) {
            output[blockIdx.x] = sharedData[0];
        }
        return;
    }

    // Tree-based reduction using unrolling for warp efficiency
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        // Prefer warp-level primitives for efficient data sharing among threads
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads(); // Wait for all threads to finish the addition
    }

    // Write the result of this block's computation to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}