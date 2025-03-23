__global__ void sumReduction(int *input, int *output, int size) {
    // Define shared memory for storing elements from input array
    extern __shared__ int sharedData[];

    // Calculate the global thread index and the local thread index within the block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // Load elements from global memory to shared memory, handling array boundaries
    sharedData[localTid] = (tid < size) ? input[tid] : 0;
    
    // Synchronize to ensure data is loaded
    __syncthreads();

    // Perform tree-based reduction using a loop
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        // Use warp-level primitive for intra-warp reduction to minimize synchronization overhead
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads(); // Necessary to ensure all threads participate fully in the addition
    }

    // Write the result of the block reduction to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}