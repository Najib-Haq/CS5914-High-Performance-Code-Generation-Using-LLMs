__global__ void sumReduction(int *input, int *output, int size) {
    // Define shared memory for storing input data per block
    extern __shared__ int sharedData[];

    // Calculate the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // Load elements into shared memory while handling array boundaries
    sharedData[localTid] = (tid < size) ? input[tid] : 0;
    
    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Perform tree-based reduction in a warp-efficient manner
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads(); // Ensure all threads have completed the addition before proceeding
    }

    // Write the result of this block's reduction to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}