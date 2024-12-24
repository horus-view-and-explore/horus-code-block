// This file contains CPU Temperature, a C++ example implementation of the Horus
// Code Block C API.
//
// Copyright (C) 2020, 2021 Horus View and Explore B.V.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../../../Horus_code_block.h"
#include "../../../Horus_cuda_code_block.h"
#include "../Helper_functions.hpp"

#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>

void video_flip_run_linear(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    unsigned char *src,
    unsigned char *dst,
    cudaStream_t &stream);

void video_flip_run_surface(
    dim3 blockDim,
    dim3 gridDim,
    size_t width,
    size_t height,
    cudaSurfaceObject_t src,
    cudaSurfaceObject_t dst,
    cudaStream_t &stream);

// =================================================================================================
// -- Context data
// --------------------------------------------------------------------------------
// =================================================================================================

struct User_context
{
    dim3 blockDim;
    dim3 gridDim;

    bool debug_schedule;

    Horus_cuda_code_block_data_buffer *src;
    Horus_cuda_code_block_data_buffer *dst;

    bool src_inited;
    bool dst_inited;

    bool src_claimed;
    bool dst_claimed;

    bool configured;
    bool use_surface;
    bool use_linear;

    void create_grid_dimensions(size_t width, size_t height)
    {
        blockDim = dim3(32, 32);
        gridDim = dim3(
            (static_cast<unsigned int>(width) + blockDim.x - 1) / blockDim.x,
            (static_cast<unsigned int>(height) + blockDim.y - 1) / blockDim.y);
    }

    void configure()
    {
        configured = true;

        use_surface = src->mem_flags & dst->mem_flags &
                      Horus_cuda_code_block_data_buffer::Memory_flags::
                          Horus_cuda_code_block_data_buffer_memory_dev_surface;

        use_linear = src->mem_flags & dst->mem_flags &
                     Horus_cuda_code_block_data_buffer::Memory_flags::
                         Horus_cuda_code_block_data_buffer_memory_dev_linear_type;
    }

    void schedule()
    {
        if (!configured)
        {
            configure();
        }

        if (use_linear)
        {
            video_flip_run_linear(
                blockDim,
                gridDim,
                src->media_width,
                src->media_height,
                (unsigned char *)src->mem_dev,
                (unsigned char *)dst->mem_dev,
                src->cuda_stream);
        }
        else if (use_surface)
        {
            video_flip_run_surface(
                blockDim,
                gridDim,
                src->media_width,
                src->media_height,
                (cudaSurfaceObject_t)src->mem_dev,
                (cudaSurfaceObject_t)dst->mem_dev,
                src->cuda_stream);
        }
    }
};

// =================================================================================================
// -- API
// ------------------------------------------------------------------------------------------
// =================================================================================================

// -- Mandatory version function
// -------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    *version = HORUS_CODE_BLOCK_VERSION;
    return Horus_code_block_success;
}

// -- Optional discovery function
// ------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    static const std::string static_discovery_info_description = "Cuda Video Flip\n";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "Cuda Video Flip: ", static_discovery_info_description.c_str()
    };
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

// -- Mandatory functions
// --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const,
    Horus_code_block_user_context **const user_context)
{
    *user_context = new User_context;
    User_context *ctx = *(reinterpret_cast<User_context **const>(user_context));
    ctx->debug_schedule = true;
    ctx->src_inited = false;
    ctx->dst_inited = false;
    ctx->src_claimed = false;
    ctx->dst_claimed = false;
    ctx->configured = false;

    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    delete static_cast<User_context *>(context->user_context); // Clean up
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    User_context *user_ctx = static_cast<User_context *>(context->user_context);

    if (data->type == Horus_cuda_code_block_buffer_info)
    {
        Horus_cuda_code_block_data_buffer *input =
            (Horus_cuda_code_block_data_buffer *)data->contents;

        switch (input->event)
        {
            // EVENT STOPPED
            case Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_event_stopped:
            {
                user_ctx->configured = false;
                if (input->index == 0)
                {
                    user_ctx->src_inited = false;
                }
                else if (input->index == 1)
                {
                    user_ctx->dst_inited = false;
                }
                break;
            }

            // EVENT INITIALIZED
            case Horus_cuda_code_block_data_buffer::
                Horus_cuda_code_block_data_buffer_event_initialized:
            {
                Helper_functions_show_buffer_info(input);
                // keep track of our buffers
                if (input->index == 0)
                {
                    user_ctx->src_inited = true;
                    user_ctx->src = input;
                    user_ctx->create_grid_dimensions(input->media_width, input->media_height);
                }
                if (input->index == 1)
                {
                    user_ctx->dst_inited = true;
                    user_ctx->dst = input;
                }
            }
            break;

            // EVENT SLOT CHANGED
            case Horus_cuda_code_block_data_buffer::
                Horus_cuda_code_block_data_buffer_event_slot_changed:
            {
                // Show the buffer state, only once
                if (user_ctx->debug_schedule)
                {
                    Helper_functions_show_buffer_state(input);
                    user_ctx->debug_schedule = false;
                }

                // if my slot_idx is the active slot, then i can use the buffer
                if (input->slot_idx == input->active_slot_idx)
                {
                    if (input->index == 0)
                    {
                        user_ctx->src_claimed = true;
                    }
                    if (input->index == 1)
                    {
                        user_ctx->dst_claimed = true;
                    }
                }

                if (user_ctx->src_claimed && user_ctx->dst_claimed)
                {
                    if (user_ctx->src_inited && user_ctx->dst_inited)
                    {
                        user_ctx->schedule();
                    }

                    // Both buffer can be released
                    input->notify_done_flags =
                        Horus_cuda_code_block_data_buffer::Notify_done_flags::
                            Horus_cuda_code_block_data_buffer_notify_done_0 |
                        Horus_cuda_code_block_data_buffer::Notify_done_flags::
                            Horus_cuda_code_block_data_buffer_notify_done_1;

                    user_ctx->src_claimed = false;
                    user_ctx->dst_claimed = false;
                }
            }

            default:
                break;
        }
    }

    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    return Horus_code_block_success;
}