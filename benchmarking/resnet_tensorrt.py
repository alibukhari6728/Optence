# benchmarking/resnet_tensorrt.py
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2
import time

# Initialize CUDA
cuda.init()

# Create CUDA context
device = cuda.Device(0)
cuda_context = device.make_context()

# Load the ONNX model
onnx_model_path = "models/resnet50.onnx"
onnx_model = onnx.load(onnx_model_path)

# Create a TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Build the TensorRT engine
def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse the ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build the engine.")
            return None
        
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized_engine)

engine = build_engine(onnx_model_path)
if engine is None:
    print("Engine could not be created.")
    exit()

# Allocate buffers
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for i in range(engine.num_bindings):
    binding = engine.get_tensor_name(i)
    size = trt.volume(engine.get_tensor_shape(binding))
    dtype = trt.nptype(engine.get_tensor_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))

# Create context
context = engine.create_execution_context()

# Prepare input data
input_image = cv2.imread("data/croc.jpeg")
input_image = cv2.resize(input_image, (224, 224))
input_image = input_image.astype(np.float32)
input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
input_image = np.ascontiguousarray(input_image)

# Copy input data to pagelocked buffer
np.copyto(inputs[0][0], input_image.ravel())

# Warm-up
for _ in range(10):
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

# Measure inference time
start_time = time.time()
for _ in range(100):
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
end_time = time.time()

print(f'TensorRT Inference Time: {(end_time - start_time) / 100:.6f} seconds')

# Validate the output
output = outputs[0][0].reshape(1, -1)
print("Output shape:", output.shape)

# Clean up
del context
del engine
cuda_context.pop()
del cuda_context
