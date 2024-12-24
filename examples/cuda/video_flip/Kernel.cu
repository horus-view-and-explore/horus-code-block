
__global__ void video_flip_kernel(unsigned char *src, unsigned char *dst, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny)
    {
        {
            size_t src_offset = y * nx * 4 + x * 4;
            size_t dst_offset = (ny - y) * nx * 4 + x * 4;

            dst[dst_offset] = src[src_offset];
            dst[dst_offset + 1] = src[src_offset + 1];
            dst[dst_offset + 2] = src[src_offset + 2];
            dst[dst_offset + 3] = src[src_offset + 3];
        }
    }
}

__global__ void video_flip_kernel(cudaSurfaceObject_t src, cudaSurfaceObject_t dst, int nx, int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny)
    {

        uchar4 data;
        surf2Dread(&data, src, x * sizeof(uchar4), y);
        surf2Dwrite(data, dst, x * sizeof(uchar4), (ny - 1) - y);
    }
}

void video_flip_run_linear(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    unsigned char *src,
    unsigned char *dst,
    cudaStream_t &stream)
{
    video_flip_kernel<<<gridDim, blockDim, 0, stream>>>(
        src, dst, static_cast<int>(width), static_cast<int>(height));
}

void video_flip_run_surface(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    cudaSurfaceObject_t src,
    cudaSurfaceObject_t dst,
    cudaStream_t &stream)
{
    video_flip_kernel<<<gridDim, blockDim, 0, stream>>>(
        src, dst, static_cast<int>(width), static_cast<int>(height));
}