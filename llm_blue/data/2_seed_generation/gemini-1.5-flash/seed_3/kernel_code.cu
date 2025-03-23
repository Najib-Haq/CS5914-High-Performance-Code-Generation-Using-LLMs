__global__ void sumReduction(int *input, int *output, int size) {
  // Optimization Strategy:
  // This kernel utilizes a hybrid approach combining warp-level reductions with a tree-based reduction in shared memory.  It prioritizes minimizing global memory accesses and maximizing warp-level parallelism.  
  // Bank conflict-free access is ensured in shared memory using modulo operations. Early exit conditions handle cases where the input size is smaller than the block size, preventing unnecessary computations.
  //  The choice of block size is a compromise between maximizing occupancy and minimizing the number of blocks for the final reduction.

  const int warpSize = 32;
  extern __shared__ int sdata[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Early exit if input size is smaller than block size.
  if (i >= size) {
    sdata[tid] = 0;
    return;
  }

  // Load data from global memory to shared memory with bank-conflict-free access.
  sdata[tid] = input[i + (tid % warpSize) * (blockDim.x / warpSize)];
  __syncthreads();


  //Warp-level reduction using __shfl_down_sync
  int sum = sdata[tid];
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  //Only the first thread of each warp writes to shared memory.
  if (tid % warpSize == 0) {
    sdata[tid / warpSize] = sum;
  }
  __syncthreads();

  //Tree-based reduction in shared memory.  Optimized for power-of-two block sizes for simplicity.
  int numWarps = blockDim.x / warpSize;
  for (int s = numWarps / 2; s > 0; s >>= 1) {
    if (tid / warpSize < s) {
      sdata[tid / warpSize] += sdata[tid / warpSize + s];
    }
    __syncthreads();
  }

  // Write the block sum to global memory
  if (tid / warpSize == 0) {
    output[blockIdx.x] = sdata[0];
  }
}