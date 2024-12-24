#include "Helper_functions.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void Helper_functions_show_buffer_info(Horus_cuda_code_block_data_buffer *input)
{
    std::stringstream ss;
    ss << "Buffer[" << input->index << "]\n";

    ss << "  memory:\n";
    ss << "    host*  :" << input->mem_host << "\n";
    ss << "    dev*   :" << input->mem_dev << "\n";
    ss << "    bytes  :" << input->mem_bytes << "\n";
    ss << "    flags  :" << input->mem_flags << "\n";

    std::vector<std::pair<Horus_cuda_code_block_data_buffer::Memory_flags, std::string>> mflags = {
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_dev_initialized,
         "Device Initialized"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_host_initialized,
         "Host Initialized"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_host_exception,
         "Exception"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_dev_array_type,
         "Array_type"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_dev_linear_type,
         "Linear_type"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_dev_texture,
         "Texture"},
        {Horus_cuda_code_block_data_buffer::Memory_flags::
             Horus_cuda_code_block_data_buffer_memory_dev_surface,
         "Surface"}};
    for (auto &fl : mflags)
    {
        if (input->mem_flags & fl.first)
        {
            ss << "      " << fl.second << "\n";
        }
    }

    ss << "  media:\n";
    ss << "    width  :" << input->media_width << "\n";
    ss << "    height :" << input->media_height << "\n";
    ss << "    codec  :" << input->media_codec << "\n";
    ss << "    pixfrmt:" << input->media_pixfmt << "\n";
    ss << "    flags  :" << input->media_flags << "\n";

    ss << "  state:\n";
    ss << "    name  :" << input->name << "\n";
    ss << "    slot  :" << input->slot_idx << "\n";
    ss << "    active:" << input->active_slot_idx << "\n";

    ss << "  cuda:\n";
    ss << "    stream  :" << input->cuda_stream << "\n";
    std::cout << ss.str() << std::endl;
}

void Helper_functions_show_buffer_state(Horus_cuda_code_block_data_buffer *input)
{
    std::stringstream ss;
    ss << "Buffer[" << input->index << "] slot changed \n";
    ss << "  state:\n";
    ss << "    slot  :" << input->slot_idx << "\n";
    ss << "    active:" << input->active_slot_idx << "\n";
    std::cout << ss.str() << std::endl;
}