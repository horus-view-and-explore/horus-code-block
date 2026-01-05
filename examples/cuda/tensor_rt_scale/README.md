# TensorRT Color Scaling with Horus Cuda Code Block

## Overview
This project implements a simple color scaling operation using TensorRT and the Horus Cuda Code Block API. It demonstrates how to process video frames using CUDA acceleration with TensorRT's scale layer for basic color transformation.

## Features
- Real-time video frame color scaling using CUDA and TensorRT
- Configurable scale, shift, and power parameters for color transformation
- Automatic buffer management between uint8 and float formats
- Support for RGBA video format
- Zero-copy GPU memory management

### Color Scaling Operation
The scale layer applies a simple color transformation:
```
output = (input * scale + shift)^power
```
Default values:
- scale: 0.2 (reduces color intensity)
- shift: 0.0 (no offset)
- power: not applied
