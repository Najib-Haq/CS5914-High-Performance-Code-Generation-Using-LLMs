__global__ void sumReduction(int *input, int *output, int size) {
    // Define the number of threads in a block
    extern __shared__ int sharedData[];

    // Calculate the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // Initialize shared memory for this block
    // Load elements from global memory to shared memory
    if (tid < size) {
        sharedData[localTid] = input[tid];
    } else {
        sharedData[localTid] = 0; // Handle out-of-bounds by assigning zero
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads(); // Wait for all threads to finish the reduction step
    }

    // Only the first thread in the block writes the result to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}