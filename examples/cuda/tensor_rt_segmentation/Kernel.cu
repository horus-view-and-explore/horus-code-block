#include "stdio.h"

__constant__ int COLORS[][3] = {
    {255, 0, 0},    // Red                  road
    {0, 255, 0},    // Green                sidewalk
    {0, 0, 255},    // Blue                 building
    {255, 255, 0},  // Yellow               wall
    {255, 0, 255},  // Magenta              fence
    {0, 255, 255},  // Cyan                 pole
    {128, 0, 0},    // Maroon               traffic light
    {0, 128, 0},    // Dark Green           traffic sign
    {0, 0, 128},    // Navy                 vegetation
    {128, 128, 0},  // Olive                terrain
    {128, 0, 128},  // Purple               sky
    {0, 128, 128},  // Teal                 person
    {255, 128, 0},  // Orange               rider
    {255, 0, 128},  // Deep Pink            car
    {128, 255, 0},  // Lime                 truck
    {0, 255, 128},  // Spring Green         bus
    {128, 0, 255},  // Electric Purple      train
    {0, 128, 255},  // Sky Blue             motorcycle
    {255, 128, 255} // Pink                 bicycle
};

// =================================================================================================
// Int float conversion
// =================================================================================================
__global__ void
convertInt8ToFloat(const unsigned char *input, float *output, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        size_t offset = y * width * 3 + x * 3;
        for (int c = 0; c < 3; c++)
        {
            output[offset + c] = input[offset + c];
            output[offset + c] -= 127.0;
        }
    }
}
__global__ void
convertFloatToInt8(const float *input, unsigned char *output, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {

        size_t dx = x;
        size_t dy = y;
        size_t dw = width;
        size_t dstride = dw * dw;
        size_t label_offset = dy * dw + dx;

        float min = 1;
        float max = 0;
        float total = 0;
        size_t max_idx = 0;
        for (size_t idx = 0; idx != 19; ++idx)
        {
            size_t orig_index = (idx * dstride) + label_offset;

            if (input[orig_index] < min)
                min = input[orig_index];

            total += input[orig_index];
            if (input[orig_index] > max)
            {
                max = input[orig_index];
                max_idx = idx;
            }
        }

        size_t offset = y * stride * 4 + x * 4;

        for (int c = 0; c < 3; c++)
        {
            output[offset + c] = COLORS[max_idx][c];
        }

        output[offset + 3] =
            (unsigned char)(255 * ((max - min) / total)); //(u_int8_t)(255 * current);
    }
}

void launch_conversion(
    unsigned char *char_buffer,
    float *float_buffer,
    int width,
    int height,
    int stride,
    cudaStream_t stream,
    bool toFloat = true)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (toFloat)
    {
        convertInt8ToFloat<<<gridSize, blockSize, 0, stream>>>(
            char_buffer, float_buffer, width, height, stride);
    }
    else
    {
        convertFloatToInt8<<<gridSize, blockSize, 0, stream>>>(
            float_buffer, char_buffer, width, height, stride);
    }
}

// =================================================================================================
// Conversion of packed to planar
// =================================================================================================
__global__ void packedToPlanarKernel(
    unsigned char *input,  // Input packed RGB (RGBRGBRGB)
    unsigned char *output, // Output planar RGB (RRR...GGG...BBB)
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int pixel_idx = y * width + x;
    const int packed_idx = pixel_idx * 3;  // Input index (packed)
    const int plane_size = width * height; // Size of one color plane

    // Read packed RGB values
    const unsigned char r = input[packed_idx];
    const unsigned char g = input[packed_idx + 1];
    const unsigned char b = input[packed_idx + 2];

    // Write to planar format
    output[pixel_idx] = r;                  // R plane
    output[pixel_idx + plane_size] = g;     // G plane
    output[plane_size * 2 + pixel_idx] = b; // B plane
}

void launch_packed_to_planar(
    unsigned char *input,
    unsigned char *output,
    size_t width,
    size_t height,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    packedToPlanarKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
}

__global__ void rearangeKernel(
    unsigned char *input,  // Input packed RGB
    unsigned char *output, // Output packed RGBA
    unsigned int *mapping,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    size_t offset_mapping = y * width + x;
    size_t offset_src = y * width * 4 + (x * 4);
    size_t offset_dst = mapping[offset_mapping] * 4;

    for (int c = 0; c < 4; c++)
    {
        output[offset_src + c] = input[offset_dst + c];
    }
}

void launch_rearange(
    unsigned char *input,
    unsigned char *output,
    unsigned int *mapping,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rearangeKernel<<<gridSize, blockSize, 0, stream>>>(input, output, mapping, width, height);
}

// Bilinear interpolation kernel for RGB to float images
__global__ void overlayImageKernel(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height)
        return;

    // Calculate scaling factors
    const float scale_x = static_cast<float>(input_width) / output_width;
    const float scale_y = static_cast<float>(input_height) / output_height;

    // Calculate source position with floating point precision
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    // Calculate integer positions for interpolation
    const int x0 = min(static_cast<int>(src_x), input_width - 1);
    const int y0 = min(static_cast<int>(src_y), input_height - 1);
    const int x1 = min(x0 + 1, input_width - 1);
    const int y1 = min(y0 + 1, input_height - 1);

    // Calculate interpolation weights
    const float wx1 = src_x - x0;
    const float wx0 = 1.0f - wx1;
    const float wy1 = src_y - y0;
    const float wy0 = 1.0f - wy1;

    // Calculate pixel positions in input image
    const int pos00 = (y0 * input_width + x0) * input_channels;
    const int pos01 = (y0 * input_width + x1) * input_channels;
    const int pos10 = (y1 * input_width + x0) * input_channels;
    const int pos11 = (y1 * input_width + x1) * input_channels;

    // Position in output image
    const int output_pos = (y * output_width + x) * output_channels;

    /* Interpolate each color channel */
    for (int c = 0; c < 3; c++)
    {
        const float val = wy0 * (wx0 * input[pos00 + c] + wx1 * input[pos01 + c]) +
                          wy1 * (wx0 * input[pos10 + c] + wx1 * input[pos11 + c]);

        unsigned char ovalue = (unsigned char)(0.5 * (output[output_pos + c])) +
                               (0.5 * static_cast<unsigned char>(val + 0.5f));
        output[output_pos + c] = ovalue;
    }
    // output[output_pos + 3] = 255;
}

// Bilinear interpolation kernel for RGB to float images
__global__ void scaleImageKernel(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height)
        return;

    // Calculate scaling factors
    const float scale_x = static_cast<float>(input_width) / output_width;
    const float scale_y = static_cast<float>(input_height) / output_height;

    // Calculate source position with floating point precision
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    // Calculate integer positions for interpolation
    const int x0 = min(static_cast<int>(src_x), input_width - 1);
    const int y0 = min(static_cast<int>(src_y), input_height - 1);
    const int x1 = min(x0 + 1, input_width - 1);
    const int y1 = min(y0 + 1, input_height - 1);

    // Calculate interpolation weights
    const float wx1 = src_x - x0;
    const float wx0 = 1.0f - wx1;
    const float wy1 = src_y - y0;
    const float wy0 = 1.0f - wy1;

    // Calculate pixel positions in input image
    const int pos00 = (y0 * input_width + x0) * input_channels;
    const int pos01 = (y0 * input_width + x1) * input_channels;
    const int pos10 = (y1 * input_width + x0) * input_channels;
    const int pos11 = (y1 * input_width + x1) * input_channels;

    // Position in output image
    const int output_pos = (y * output_width + x) * output_channels;

    /* Interpolate each color channel */
    for (int c = 0; c < output_channels; c++)
    {
        const float val = wy0 * (wx0 * input[pos00 + c] + wx1 * input[pos01 + c]) +
                          wy1 * (wx0 * input[pos10 + c] + wx1 * input[pos11 + c]);

        output[output_pos + c] = static_cast<unsigned char>(val + 0.5f);
    }
}

void launch_scale_image(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels,
    cudaStream_t stream)
{
    int width = input_width;
    int height = input_height;
    if ((input_width * input_height) < (output_width * output_height))
    {
        width = output_width;
        height = output_height;
    }

    // Use 16x16 thread blocks
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    scaleImageKernel<<<gridSize, blockSize, 0, stream>>>(
        input,
        output,
        input_width,
        input_height,
        input_channels,
        output_width,
        output_height,
        output_channels);
}

void launch_overlay_image(
    const unsigned char *input,
    unsigned char *output,
    int input_width,
    int input_height,
    int input_channels,
    int output_width,
    int output_height,
    int output_channels,
    cudaStream_t stream)
{
    int width = input_width;
    int height = input_height;
    if ((input_width * input_height) < (output_width * output_height))
    {
        width = output_width;
        height = output_height;
    }

    // Use 16x16 thread blocks
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    overlayImageKernel<<<gridSize, blockSize, 0, stream>>>(
        input,
        output,
        input_width,
        input_height,
        input_channels,
        output_width,
        output_height,
        output_channels);
}