__global__ void
convertInt8ToFloat(const unsigned char *input, float *output, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        size_t offset = y * width * 4 + x * 4;
        for (int c = 0; c < 4; c++)
        {
            output[offset + c] = static_cast<unsigned char>(input[offset + c]);
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
        size_t offset = y * width * 4 + x * 4;
        for (int c = 0; c < 4; c++)
        {
            float val = input[offset + c];
            val = min(max(val, 0.0f), 255.0f); // Clamp to valid range
            output[offset + c] = static_cast<unsigned char>(val);
        }
    }
}

void launchConversion(
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