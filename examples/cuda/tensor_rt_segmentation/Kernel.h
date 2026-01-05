#ifndef KERNEL_H
#define KERNEL_H

void launch_scale_image(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels,
    cudaStream_t stream);

void launch_overlay_image(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels,
    cudaStream_t stream);

void launch_conversion(
    unsigned char *char_buffer,
    float *float_buffer,
    int width,
    int height,
    int stride,
    cudaStream_t stream,
    bool toFloat = true);

void launch_packed_to_planar(
    unsigned char *input,
    unsigned char *output,
    size_t width,
    size_t height,
    cudaStream_t stream);

void launch_rearange(
    unsigned char *input,
    unsigned char *output,
    unsigned int *mapping,
    int width,
    int height,
    cudaStream_t stream);

#endif