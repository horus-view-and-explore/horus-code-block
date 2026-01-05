#ifndef HORUS_CUDA_HPP
#define HORUS_CUDA_HPP

#include <string>
#include <vector>

#include "../../Horus_cuda_code_block.h"

namespace hrs {

class HC_buffer
{
  public:
    enum HC_flags
    {
        HC_flags_free_memory = 1 << 0,
        HC_flags_has_resolution = 1 << 1,
        HC_flags_memory_linear = 1 << 2,
        HC_flags_memory_array = 1 << 3,
        HC_flags_memory_texture = 1 << 4,
        HC_flags_initialized = 1 << 5,
    };

    HC_buffer(Horus_cuda_code_block_data_buffer *buffer);
    HC_buffer(
        const std::string &name,
        size_t chantype,
        size_t numchan,
        size_t width,
        size_t height,
        const std::string &pixfmt,
        const std::string &codec);

    ~HC_buffer();

    bool download(cudaStream_t stream, bool sync = false);
    bool upload(cudaStream_t stream, bool sync = false);
    bool initialized();
    void save(const std::string &path, bool autofilename = true);

    std::string info();

    size_t flags;
    size_t width;
    size_t height;
    size_t bytes;
    size_t channels;
    size_t channel_format;

    std::string name;
    std::string codec;
    std::string pixfmt;

    std::vector<unsigned char> host;

    void *dev;
};

} // namespace hrs
#endif // HORUS_CUDA_HPP
