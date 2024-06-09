import onnxruntime as ort
import numpy as np
import time

# Check available providers
print("Available providers:", ort.get_available_providers())

# Load the ONNX model with GPU (CUDA) execution provider if available
onnx_model_path = "models/resnet50.onnx"
providers = ['CUDAExecutionProvider'] #if 'CUDAExecutionProvider' in ort.get_available_providers()# else ['CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)

# Prepare input data
input_image = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warm-up
for _ in range(10):
    _ = session.run(None, {"input": input_image})

# Measure inference time
start_time = time.time()
for _ in range(100):
    _ = session.run(None, {"input": input_image})
end_time = time.time()

print(f'ONNX Runtime Inference Time: {(end_time - start_time) / 100:.6f} seconds')