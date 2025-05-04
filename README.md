# CS5914-High-Performance-Code-Generation-Using-LLMs

Project Name: Tensor Reduction utilizing GPU <br>
This is a course project under CS 5914: High-Performance Code Generation Using LLMs. The objective of this project is to optimize reduction kernels both manually and through LLMs, and try to compare performance of the two approaches in GPU accelerators. 

## Team Information
- Red team (manual optimization): Najibul Haque Sarker
- Blue team (optimization via LLMs): Prayash Joshi

## Folder Structure
- [`llm_blue`](./llm_blue/) folder contains all the code and scripts required to reproduce results for the blue team (optimization via LLMs). 
- [`manual_red`](./manual_red/) folder contains all the code and scripts required to reproduce results for the red team (manual optimization).

## Setup & Run Code
- [`llm_blue/README.md`](./llm_blue/README.md) file has setup instructions for the blue team (optimization via LLMs).  
- [`manual_red/README.md`](./manual_red/README.md) file has setup instructions for the red team (manual optimization).

### GPU specifications:
The evaluation is done on the following GPU specifications:
- NVIDIA A100-SXM4 GPU
- VRAM 81920 MB
- CUDA Version 12.2