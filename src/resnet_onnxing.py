# resnet_model.py
import torch
import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet model
resnet50 = models.resnet50(pretrained=True)

# Switch model to evaluation mode
resnet50.eval()

# Example input tensor
example_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(resnet50, example_input, "resnet50.onnx", 
                  export_params=True, opset_version=11, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'])

print("Model exported to resnet50.onnx")
