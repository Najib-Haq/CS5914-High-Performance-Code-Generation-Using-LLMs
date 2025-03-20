# CS5914-High-Performance-Code-Generation-Using-LLMs

Tensor Reduction in GPU (manual optimization)

# Reduction Strategies
## Strategy 1: Naive Parallel Implementation

File: `parallel_reduction1.cu` <br> 
Logic: A straightforward tree-based reduction where each pair of elements is processed by one thread. This naive kernel loops over reduction steps, at each step having certain threads add a neighbor’s value to their own. For example, thread 0 adds thread 1’s element, thread 2 adds thread 3’s, and so on, halving the number of active threads each iteration. A barrier sync (__syncthreads()) is used between steps to ensure partial sums are visible to the next step

## Strategy 2: Sequential Addressing + Shared memory

File: `parallel_reduction2.cu` <br> 
Logic: Alter the reduction loop to use sequential addressing.  In sequential addressing, threads with index below a certain threshold perform the addition with a partner thread above the threshold. This way, in each step, the lower half of threads in the block actively add the values of the upper half, eliminating warp-level divergence because all threads in a given warp either all pass the condition or all fail it as a group. The access pattern becomes more regular: thread 0 adds thread s, thread 1 adds thread s+1, etc., which are contiguous in memory.

## Strategy 3: Unrolling the Final Warp Loop + shuffle instructions

File: `parallel_reduction3.cu` <br> 
Logic: Final stages of the reduction (when only a warp or less of threads remain) still rely on shared memory and __syncthreads. Once we get down to 32 or fewer elements, those reside within a single warp. Intra-warp communication can be done more efficiently using warp shuffle instructions rather than shared memory. Another optimization is to fully unroll the reduction for the last warp of 32 threads using manual code.


## Strategy 4: Fully Unrolled Reduction Kernel 

File: `parallel_reduction4.cu` <br> 
Logic: Using C++ templates or launch-time constants to fully unroll the entire reduction for a given block size. The idea is to eliminate all loop overhead and make every memory access pattern a compile-time decision, which allows the compiler to optimize and schedule instructions most effectively.


## Run Code
In CLI:
```
nvcc file_name.cu -o object_name
./object_name
```

<!-- nvcc your_cuda_program.cu -o output_binary -lnvidia-ml -->


## Profiling
- Basic profiling: `ncu ./object`
- Profiling with thread divergence benchmarks: `ncu --set full ./object`
- Wrap Execution Efficiency: `ncu --metrics smsp__thread_inst_executed_per_inst_executed ./object`
- FLOPS (very simple here due to addition operation): `ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on,smsp__sass_thread_inst_executed_op_fmul_pred_on,smsp__sass_thread_inst_executed_op_ffma_pred_on ./object`
- Memory Usage: `ncu --metrics dram__bytes_read.sum,l1tex__t_bytes.sum,l2tex__t_bytes.sum ./object` 


## Links
- ncu commands: https://gist.github.com/getianao/1686c4d0dac02a0b91a2885e18d9c9a3 