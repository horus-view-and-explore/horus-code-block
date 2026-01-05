#ifndef HORUS_CUDA_CODE_BLOCK_H
#define HORUS_CUDA_CODE_BLOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

/// Version of this Horus Code Block C API.
#define HORUS_CUDA_CODE_BLOCK_VERSION 0u

struct Horus_cuda_code_block_data_buffer
{
    enum Event_type
    {
        Horus_cuda_code_block_data_buffer_event_initialized,
        Horus_cuda_code_block_data_buffer_event_slot_changed,
        Horus_cuda_code_block_data_buffer_event_stopped,
    };

    enum Memory_flags
    {
        Horus_cuda_code_block_data_buffer_memory_dev_initialized = 1 << 0,
        Horus_cuda_code_block_data_buffer_memory_host_initialized = 1 << 1,
        Horus_cuda_code_block_data_buffer_memory_host_exception = 1 << 2,
        Horus_cuda_code_block_data_buffer_memory_dev_array_type = 1 << 3,
        Horus_cuda_code_block_data_buffer_memory_dev_linear_type = 1 << 4,
        Horus_cuda_code_block_data_buffer_memory_dev_texture = 1 << 20,
        Horus_cuda_code_block_data_buffer_memory_dev_surface = 1 << 21
    };

    enum Media_flags
    {
        Horus_cuda_code_block_data_buffer_media_width = 1 << 0,
        Horus_cuda_code_block_data_buffer_media_height = 1 << 1,
        Horus_cuda_code_block_data_buffer_media_codec = 1 << 2,
        Horus_cuda_code_block_data_buffer_media_pixfmt = 1 << 3,
    };

    enum Notify_done_flags
    {
        Horus_cuda_code_block_data_buffer_notify_done_0 = 1 << 0,
        Horus_cuda_code_block_data_buffer_notify_done_1 = 1 << 1,
    };

    // Memory section
    void *mem_dev;    // linear,cudaArray_t,cudaSurfaceObject_t,cudaTextureObject_t
    void *mem_host;   // host memory
    size_t mem_bytes; // nr of bytes
    size_t mem_flags; // configuration flags

    // Media section
    size_t media_width;
    size_t media_height;
    std::string media_codec;  // Codec
    std::string media_pixfmt; // Pixel format
    size_t media_flags;       // flags for which fields have been set

    size_t index;     // which external buffer idx
    size_t slot_idx;  // slot id that has been granted to us
    std::string name; // buffer name

    // Events
    Event_type event;         // reason event
    size_t active_slot_idx;   // current active slot
    size_t notify_done_flags; // signals the code block that we are done with a specific buffer
                              // on a specific input buffer multiple buffer done flags can be set.

    // Cuda types
    cudaStream_t cuda_stream; // runn jobs on this stream
};

inline bool Horus_cuda_code_block_memcpy_async(
    void *&mem,
    void *data,
    size_t size,
    cudaStream_t &cuda_stream,
    cudaMemcpyKind kind)
{
    switch (kind)
    {
        case cudaMemcpyKind::cudaMemcpyHostToDevice:
            cudaMemcpyAsync(mem, data, size, cudaMemcpyHostToDevice, cuda_stream);
            break;

        case cudaMemcpyKind::cudaMemcpyDeviceToHost:
            cudaMemcpyAsync(data, mem, size, cudaMemcpyDeviceToHost, cuda_stream);
        default:
        {
        }
        break;
    }

    cudaStreamSynchronize(cuda_stream);
    return cudaGetLastError() == cudaSuccess;
}

inline bool Horus_cuda_code_block_memcpy_chars(
    void *&mem,
    void *data,
    size_t size,
    cudaStream_t &cuda_stream,
    cudaMemcpyKind kind)
{
    return Horus_cuda_code_block_memcpy_async(mem, data, size, cuda_stream, kind);
}

inline bool Horus_cuda_code_block_memcpy_floats(
    void *&mem,
    void *data,
    size_t size,
    cudaStream_t &cuda_stream,
    cudaMemcpyKind kind)
{
    return Horus_cuda_code_block_memcpy_async(mem, data, size * sizeof(float), cuda_stream, kind);
}

inline void Horus_cuda_code_block_show_error(const char *info)
{
    std::cout << "Info: " << info << std::endl;
    std::cout << "Nr:   " << cudaPeekAtLastError() << std::endl;
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
}

inline void Horus_cuda_code_block_synchronize(cudaStream_t stream, const char *info)
{
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        Horus_cuda_code_block_show_error("PRE SYNC");
        Horus_cuda_code_block_show_error(info);
        exit(1);
    }

    cudaStreamSynchronize(stream);

    if (cudaPeekAtLastError() != cudaSuccess)
    {
        Horus_cuda_code_block_show_error(info);
        exit(1);
    }
}

#endif // HORUS_CUDA_CODE_BLOCK_H
