#include "Horus_cuda.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace hrs {

HC_buffer::HC_buffer(Horus_cuda_code_block_data_buffer *buffer)
    : flags(0)
    , width(0)
    , height(0)
    , bytes(buffer->mem_bytes)
    , channels(0)
    , channel_format(0)
    , name(buffer->name)
    , codec(buffer->media_codec)
    , pixfmt(buffer->media_pixfmt)
    , dev(buffer->mem_dev)
{
    size_t resolution =
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_media_width &&
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_media_height;

    if (buffer->mem_flags &
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_memory_dev_initialized)
    {
        flags |= HC_flags::HC_flags_initialized;
    }
    else
    {
        return;
    }

    if (buffer->media_flags & resolution)
    {
        flags |= HC_flags::HC_flags_has_resolution;
        width = buffer->media_width;
        height = buffer->media_height;
    }

    if (buffer->mem_flags &
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_memory_dev_array_type)
    {
        flags |= HC_flags::HC_flags_memory_array;
    }
    else if (
        buffer->mem_flags &
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_memory_dev_linear_type)
    {
        flags |= HC_flags::HC_flags_memory_linear;
    }
    else if (
        buffer->mem_flags &
        Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_memory_dev_texture)
    {
        flags |= HC_flags::HC_flags_memory_texture;
    }

    if (codec == "RAWVIDEO")
    {
        if (pixfmt == "RGB24")
            channels = 3;
        else if (pixfmt == "RGBA")
            channels = 4;
    }
    //@todo
    channel_format = 1;
}

HC_buffer::HC_buffer(
    const std::string &n,
    size_t chantype,
    size_t numchan,
    size_t w,
    size_t h,
    const std::string &pxfmt,
    const std::string &cdc)
    : flags(0)
    , width(w)
    , height(h)
    , bytes(w * h * numchan * chantype)
    , channels(numchan)
    , channel_format(chantype)
    , name(n)
    , codec(cdc)
    , pixfmt(pxfmt)
    , dev(nullptr)
{
    flags |= HC_flags::HC_flags_has_resolution;
    flags |= HC_flags::HC_flags_memory_linear;

    cudaMalloc(&dev, bytes);
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error [cudaMalloc]: " << name << "\n";
        std::cout << cudaGetErrorString(error) << std::endl;
    }
    else
    {
        flags |= HC_flags::HC_flags_initialized;
        flags |= HC_flags::HC_flags_free_memory;
    }
}

HC_buffer::~HC_buffer()
{
    if (flags & HC_flags_free_memory)
    {
        cudaError error = cudaFree(dev);

        if (error != cudaSuccess)
        {
            std::cout << "CUDA error [cudaFree]: " << name << "\n";
            std::cout << cudaGetErrorString(error) << std::endl;
        }
    }
}

bool HC_buffer::download(cudaStream_t stream, bool sync)
{
    host.resize(bytes);
    cudaMemcpyAsync(host.data(), dev, bytes, cudaMemcpyDeviceToHost, stream);

    if (sync)
    {
        cudaStreamSynchronize(stream);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool HC_buffer::upload(cudaStream_t stream, bool sync)
{
    if (host.size() != bytes)
        return false;

    cudaMemcpyAsync(dev, host.data(), bytes, cudaMemcpyHostToDevice, stream);

    if (sync)
    {
        cudaStreamSynchronize(stream);
    }
    return cudaGetLastError() == cudaSuccess;
}

void HC_buffer::save(const std::string &path, bool autofilename)
{
    std::stringstream ss;
    ss << path;

    if (autofilename)
    {
        if (!name.empty())
            ss << name;

        if (flags & HC_flags::HC_flags_has_resolution)
            ss << "-" << width << "x" << height;

        if (!pixfmt.empty())
            ss << "-" << pixfmt;

        if (!codec.empty())
            ss << "." << codec;
    }

    std::cout << "saving: " << ss.str() << std::endl;
    std::ofstream file(ss.str(), std::ios::out | std::ios::binary);
    std::copy(host.cbegin(), host.cend(), std::ostream_iterator<unsigned char>(file));
}

bool HC_buffer::initialized()
{
    return flags & HC_flags_initialized;
}

std::string HC_buffer::info()
{
    std::stringstream ss;
    ss << "\n******************\n";
    ss << "**    HC_buffer   **\n";
    ss << "name           :" << name << "\n";
    ss << "dev            :" << dev << "\n";
    ss << "flags          :" << flags << "\n";

    std::vector<std::pair<HC_flags, std::string>> bflags = {
        {HC_flags::HC_flags_free_memory, "free_memory"},
        {HC_flags::HC_flags_has_resolution, "has_resolution"},
        {HC_flags::HC_flags_memory_linear, "memory_linear"},
        {HC_flags::HC_flags_memory_array, "memory_array"},
        {HC_flags::HC_flags_memory_texture, "memory_texture"}};

    for (auto &fl : bflags)
    {
        if (flags & fl.first)
        {
            ss << "  " << fl.second << "\n";
        }
    }

    ss << "width          :" << width << "\n";
    ss << "height         :" << height << "\n";
    ss << "bytes          :" << bytes << "\n";
    ss << "channels       :" << channels << "\n";
    ss << "channel_format :" << channel_format << "\n";
    ss << "codec          :" << codec << "\n";
    ss << "pixfmt         :" << pixfmt << "\n";
    ss << "******************\n";

    return ss.str();
}

} // namespace hrs
