# CS3914-High-Performance-Code-Generation-Using-LLMs

Tensor Reduction in GPU (manual optimization)

- Parallel Reduction v1
## Run Code
In CLI:
```
nvcc file_name.cu -o object_name
./object_name
```


## Profiling
- Basic profiling: `ncu ./object`
- Profiling with thread divergence benchmarks: `ncu --set full ./object`
- FLOPS (very simple here due to addition operation): `ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on,smsp__sass_thread_inst_executed_op_fmul_pred_on,smsp__sass_thread_inst_executed_op_ffma_pred_on ./object`
- Memory Usage: `ncu --metrics dram__bytes_read.sum,l1tex__t_bytes.sum,l2tex__t_bytes.sum ./object` 