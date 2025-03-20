# CS3914-High-Performance-Code-Generation-Using-LLMs

Tensor Reduction in GPU (manual optimization)

- Parallel Reduction v1
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