# Optence
AI Inference Frameworks; Custom CUDA kernels; and Nsight Suite.

### Part 1: OptCNN

Optimizing Inference Performance for a Deep Convolutional Neural Network, supported by profiling with Nsight-Systems.

Target Frameworks:
- PyTorch 
- ONNX Runtime
- TensorRT

### Part 2: OptLLM

Optimizing Inference Performance for a Transformer-based Large Language Model, supported by profiling with Nsight-Systems.

Target Frameworks:
- PyTorch 
- ONNX Runtime
- TensorRT
- TensorRT-LLM

### Part 3: Custom-CUDA Kernels

Optimizing Inference Performance for a CNN and/or a Transformer.

Goals:
- Select a model component, replace it with a custom CUDA kernel (e.g. a convolutional kernel).
- Hyper-optimize the custom kernel to target hardware and problem size.
- Build and test various versions of the same kernel, with increasingly better memory and kernel management.
- Profile using Nsight-Systems AND Nsight-Compute to achieve best possible kernel.

### Part 4: Custom-CUDA Model

Build a custom neural network in CUDA & C++ from scratch. 

Goals:
- Implement a neural network in CUDA & C++.
- Hyperoptimize its operations for inference time.
- Apply advanced concepts in kernel optimization, as well as good memory access patterns.
- Profile using Nsight-Systems AND Nsight-Compute to achieve best possible kernel.

-----------------------------------------------------------------------------------------------

## Sequence of Delivery:


[X] Part 1: OptCNN

[ ] Part 4: Custom-CUDA Model ( ETA 14/06/2024)

[ ] Part 2: OptLLM  ( ETA 16/06/2024)

[ ] Part 3: Custom-CUDA Kernels ( ETA 17/06/2024)
