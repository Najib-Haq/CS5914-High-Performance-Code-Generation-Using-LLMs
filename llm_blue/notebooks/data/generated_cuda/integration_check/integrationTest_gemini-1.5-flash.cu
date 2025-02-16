#include <stdio.h>

__global__ void helloFromCUDA(int threads) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < threads) {
    printf("Hello from CUDA thread %d!\n", i);
  }
}


int main() {
  int numThreads = 256;
  int threadsPerBlock = 256;
  int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;


  if (numThreads <= 0) {
    printf("Error: Number of threads must be positive.\n");
    return 1;
  }

  if (threadsPerBlock <=0){
    printf("Error: Threads per block must be positive.\n");
    return 1;
  }

  helloFromCUDA<<<blocksPerGrid, threadsPerBlock>>>(numThreads);

  cudaDeviceSynchronize(); // Wait for kernel to complete

  return 0;
}