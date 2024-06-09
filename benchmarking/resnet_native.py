import torch
import torchvision.models as models
import time

# Load the PyTorch model
model = models.resnet50(pretrained=True).eval().cuda()

# Prepare input data
input_image = torch.randn(1, 3, 224, 224).cuda()

# Warm-up
for _ in range(10):
    _ = model(input_image)

# Measure inference time
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(input_image)
end_time = time.time()

print(f'PyTorch Inference Time: {(end_time - start_time) / 100:.6f} seconds')