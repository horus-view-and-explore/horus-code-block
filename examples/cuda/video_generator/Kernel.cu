
__global__ void video_gen_kernel(unsigned char *buffer, int nx, int ny, unsigned int count)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny)
    {
        {
            uchar4 data = make_uchar4((x + count * 4) % 255, (y + count * 4) % 255, 100, 255);
            size_t offset = y * nx * 4 + x * 4;
            buffer[offset] = data.x;
            buffer[offset + 1] = data.y;
            buffer[offset + 2] = data.z;
            buffer[offset + 3] = data.w;
        }
    }
}

__global__ void video_gen_kernel(cudaSurfaceObject_t surface, int nx, int ny, unsigned int count)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny)
    {
        {
            uchar4 data = make_uchar4((x + count * 4) % 255, (y + count * 4) % 255, 100, 255);
            surf2Dwrite(data, surface, x * sizeof(uchar4), y);
        }
    }
}

void video_gen_run_linear(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    unsigned char *buffer,
    cudaStream_t &stream,
    size_t count)
{
    video_gen_kernel<<<gridDim, blockDim, 0, stream>>>(
        buffer, static_cast<int>(width), static_cast<int>(height), count);
}

void video_gen_run_surface(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    cudaSurfaceObject_t so,
    cudaStream_t &stream,
    size_t count)
{
    video_gen_kernel<<<gridDim, blockDim, 0, stream>>>(
        so, static_cast<int>(width), static_cast<int>(height), count);
}