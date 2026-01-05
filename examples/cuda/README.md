# Horus CUDA Code Block
The Horus CUDA Code Block extends CUDA programming capabilities to the base Horus Code Block system. Developers need to include both `Horus_code_block.h` and `Horus_cuda_code_block.h` header files to use this functionality.

## Examples

### 1. Video Test Source Generator
Demonstrates basic single-buffer video manipulation.

### 2. Enhanced Video Processing with Flip
Builds on Example 1 by implementing a dual-buffer approach for video manipulation, adding flip functionality.

### 3. TensorRT Scale Layer
Shows how to create a TensorRT network with a Scale layer that transforms input video using the formula:
```
output = (input * scale + shift)^(power)
```

### 4. TensorRT-Based Image Segmentation
Demonstrates image segmentation using TensorRT with these key features:
- Loads a pre-trained ResNet101_DUC_HDC model from ONNX format
- Performs image segmentation on the input
- Generates an overlay of the segmentation results
