__global__ void sumReduction(int *input, int *output, int size) {
  // Optimization Strategy:
  // This kernel uses a warp-level reduction using __shfl_down_sync() before the shared memory reduction. This significantly reduces the number of shared memory accesses and synchronization points, improving performance, especially for large datasets.
  // It also employs bank-conflict-free memory access by ensuring that threads within a warp access different memory banks.  This minimizes memory access latency and improves overall throughput.
  // The number of threads per block is chosen to be a multiple of 32 (warp size) to leverage warp-level primitives effectively.


  extern __shared__ int sdata[]; // Allocate shared memory dynamically

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Load data from global memory to shared memory, using bank-conflict free access pattern
  if (i < size) {
    sdata[tid] = input[i + tid % 32 * (blockDim.x / 32) ]; //Bank conflict free access if blockDim.x is multiple of 32
  } else {
    sdata[tid] = 0; 
  }
  __syncthreads();

  // Warp-level reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

    //Further reduction within the warp using warp-level primitives
    if(tid < 32){
        int val = sdata[tid];
        for(int offset = 16; offset > 0; offset >>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(tid == 0) sdata[0] = val;
    }
  __syncthreads();


  // Write the block sum to global memory
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}