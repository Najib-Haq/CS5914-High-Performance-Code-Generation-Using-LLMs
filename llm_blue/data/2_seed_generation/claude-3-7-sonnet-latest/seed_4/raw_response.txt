```cuda
__global__ void sumReduction(int *input, int *output, int size) {
    // Optimization strategy:
    // 1. Use sequential addressing to avoid shared memory bank conflicts
    // 2. Process 4 elements per thread during loading to increase arithmetic intensity
    // 3. First reduction during loading phase to reduce total reduction steps
    // 4. Grid-stride loop for handling arbitrarily large arrays efficiently
    // 5. Full unrolling of the final warp reduction using shuffle operations
    // 6. Minimize divergent execution with structured conditionals
    // 7. Efficient boundary checking only where necessary
    // 8. Avoid unnecessary synchronization in the last warp
    // 9. Compile-time loop unrolling for critical reduction path
    // 10. Staggered reduction pattern to maximize parallel operations

    extern __shared__ int sdata[];
    
    // Thread identification
    const unsigned int tid = threadIdx.x;
    const unsigned int blockSize = blockDim.x;
    const unsigned int globalIdx = blockIdx.x * blockDim.x * 4 + tid; // *4 because each thread loads 4 elements
    const unsigned int gridSize = blockDim.x * gridDim.x * 4;         // *4 for the stride
    
    // Initialize thread-local sum
    int thread_sum = 0;
    
    // Grid-stride loop - each thread processes 4 elements per iteration
    for (unsigned int i = globalIdx; i < size; i += gridSize) {
        thread_sum += input[i];
        
        // Check and add second element (i + blockDim.x)
        if (i + blockDim.x < size) {
            thread_sum += input[i + blockDim.x];
            
            // Check and add third element (i + 2*blockDim.x)
            if (i + 2 * blockDim.x < size) {
                thread_sum += input[i + 2 * blockDim.x];
                
                // Check and add fourth element (i + 3*blockDim.x)
                if (i + 3 * blockDim.x < size) {
                    thread_sum += input[i + 3 * blockDim.x];
                }
            }
        }
    }
    
    // Store thread sum to shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree-based reduction in shared memory with sequential addressing
    // Reduces shared memory bank conflicts and improves performance
    
    // Unroll the loop for blockSize >= 1024 (compile-time optimization)
    if (blockSize >= 1024 && tid < 512) {
        sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    
    if (blockSize >= 512 && tid < 256) {
        sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    
    if (blockSize >= 256 && tid < 128) {
        sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    
    if (blockSize >= 128 && tid < 64) {
        sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Final warp reduction using warp shuffle (no shared memory access or sync needed)
    if (tid < 32) {
        // For blockSize >= 64, we need to add from shared memory
        if (blockSize >= 64) {
            sdata[tid] += sdata[tid + 32];
        }
        
        // Use warp shuffle operations for the final reduction steps (fully unrolled)
        int value = sdata[tid];
        
        // Using full-mask for maximum compatibility across different warp sizes
        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value +=