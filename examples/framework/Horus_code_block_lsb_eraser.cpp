// This file is part of the C++ example implementations of the Horus Code Block C API.
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

#include "../../Horus_code_block.h"

#include "Horus_code_block_output_message_base.hpp"
#include "Horus_code_block_user_context_base.hpp"

#include <cassert>
#include <cstring>
#include <list>
#include <map>
#include <queue>
#include <string>

namespace {

// =================================================================================================
// -- Internal data --------------------------------------------------------------------------------
// =================================================================================================

const std::string _video_output_pipe_key_("video-output-pipe");

// =================================================================================================
// -- Internal types -------------------------------------------------------------------------------
// =================================================================================================

class Video_output_message : public Horus_code_block_output_message_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    Video_output_message(const Horus_code_block_output_pipe output_pipe)
        : Horus_code_block_output_message_base(output_pipe)
        , _data_video_fourcc_in_queue()
        , _data_video_fourcc_out{}
        , _buffer_request_id_to_data_video_fourcc_in_map()
    {
    }

    // =============================================================================================
    // -- Horus_code_block_output_message_base -----------------------------------------------------
    // =============================================================================================

    virtual bool data_available() const final override
    {
        return (
            Horus_code_block_output_message_base::data_available() ||
            data_video_fourcc_out_available());
    }

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    bool data_video_fourcc_out_available() const
    {
        return !_data_video_fourcc_in_queue.empty();
    }

    void push_data_video_fourcc_out(
        const struct Horus_code_block_data_video_fourcc_in &data_video_fourcc_in)
    {
        assert(
            data_output_message_begin_available() &&
            _buffer_request_id_to_data_video_fourcc_in_map.empty());
        _data_video_fourcc_in_queue.push(data_video_fourcc_in);
    }

    const struct Horus_code_block_data_video_fourcc_out &pop_data_video_fourcc_out()
    {
        assert(!data_output_message_begin_available() && data_video_fourcc_out_available());
        {
            const struct Horus_code_block_data_video_fourcc_in &data_video_fourcc_in =
                _data_video_fourcc_in_queue.front();

            _data_video_fourcc_out.data_id = nullptr;
            _data_video_fourcc_out.buffer_request_id =
                _buffer_request_id_to_data_video_fourcc_in_map.size();
            _data_video_fourcc_out.buffer_size = data_video_fourcc_in.buffer_size;
            std::memcpy(_data_video_fourcc_out.fourcc, data_video_fourcc_in.fourcc, 4);
            _data_video_fourcc_out.width = data_video_fourcc_in.width;
            _data_video_fourcc_out.height = data_video_fourcc_in.height;
            _data_video_fourcc_out.is_key_frame = data_video_fourcc_in.is_key_frame;
            _data_video_fourcc_out.line_stride = data_video_fourcc_in.line_stride;

            _buffer_request_id_to_data_video_fourcc_in_map.emplace(
                _data_video_fourcc_out.buffer_request_id, data_video_fourcc_in);
        }
        _data_video_fourcc_in_queue.pop();
        return _data_video_fourcc_out;
    }

    void write_buffer(const struct Horus_code_block_data_buffer &data_buffer)
    {
        const auto &buffer_request_id_and_data_video_fourcc_in_pair =
            _buffer_request_id_to_data_video_fourcc_in_map.find(data_buffer.buffer_request_id);
        assert(
            buffer_request_id_and_data_video_fourcc_in_pair !=
            _buffer_request_id_to_data_video_fourcc_in_map.end());
        {
            const struct Horus_code_block_data_video_fourcc_in &data_video_fourcc_in =
                buffer_request_id_and_data_video_fourcc_in_pair->second;
            assert(data_video_fourcc_in.buffer_size == data_buffer.buffer_size);
            const char *input = data_video_fourcc_in.buffer;
            const char *const input_end = input + data_video_fourcc_in.buffer_size;
            char *output = data_buffer.buffer;
            while (input < input_end)
            {
                *output = *input & 0xF0;
                ++input;
                ++output;
            }
        }
        _buffer_request_id_to_data_video_fourcc_in_map.erase(
            buffer_request_id_and_data_video_fourcc_in_pair);
    }

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    std::queue<
        Horus_code_block_data_video_fourcc_in,
        std::list<Horus_code_block_data_video_fourcc_in>>
        _data_video_fourcc_in_queue;

    struct Horus_code_block_data_video_fourcc_out _data_video_fourcc_out;

    std::map<Horus_code_block_request_id, Horus_code_block_data_video_fourcc_in>
        _buffer_request_id_to_data_video_fourcc_in_map;
};

class User_context final : public Horus_code_block_user_context_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    User_context(const struct Horus_code_block_key_value_array &key_value_array)
        : Horus_code_block_user_context_base()
        , _video_output_message(get_data_output_pipe_or(
              key_value_array,
              _video_output_pipe_key_,
              Horus_code_block_output_pipe_data_0))
    {
        push_log(Horus_code_block_log_severity_info, "Opened.");
        /// Subscribe the Data 0 input pipe to FourCC video.
        push_subscriptions(
            {Horus_code_block_input_pipe_data_0}, {Horus_code_block_data_type_video_fourcc_in});
        /// Subscribe the other data input pipes no none.
        push_subscriptions(
            {Horus_code_block_input_pipe_data_1, Horus_code_block_input_pipe_data_2}, {});
    }

    // =============================================================================================
    // -- Horus_code_block_user_context_base -------------------------------------------------------
    // =============================================================================================

    // -- Data -------------------------------------------------------------------------------------

  private:
    virtual bool data_available() const final override
    {
        return (
            Horus_code_block_user_context_base::data_available() ||
            _video_output_message.data_available());
    }

    virtual void get_first_available_data(const struct Horus_code_block_data *&data) final override
    {
        if (_video_output_message.data_output_message_begin_available())
        {
            data = get_data(_video_output_message.get_data_output_message_begin());
        }
        else if (_video_output_message.data_video_fourcc_out_available())
        {
            data = get_data(_video_output_message.pop_data_video_fourcc_out());
        }
        else
        {
            Horus_code_block_user_context_base::get_first_available_data(data);
        }
    }

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

  public:
    Horus_code_block_result handle_write(const struct Horus_code_block_data &data)
    {
        switch (data.type)
        {
            case Horus_code_block_data_type_start:
                push_log(Horus_code_block_log_severity_info, "Started.");
                break;
            case Horus_code_block_data_type_stop:
                push_log(Horus_code_block_log_severity_info, "Stopped.");
                break;
            case Horus_code_block_data_type_input_message_begin:
                assert(!_video_output_message.data_available());
                break;
            case Horus_code_block_data_type_video_fourcc_in:
                _video_output_message.ensure_data_output_message_begin();
                assert(data.contents != nullptr);
                _video_output_message.push_data_video_fourcc_out(
                    *static_cast<const struct Horus_code_block_data_video_fourcc_in *>(
                        data.contents));
                break;
            case Horus_code_block_data_type_buffer:
                assert(data.contents != nullptr);
                _video_output_message.write_buffer(
                    *static_cast<const struct Horus_code_block_data_buffer *>(data.contents));
                break;
            default:;
        }
        return Horus_code_block_success;
    }

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    Video_output_message _video_output_message;
};

} // namespace

// =================================================================================================
// -- API ------------------------------------------------------------------------------------------
// =================================================================================================

// -- Mandatory version function -------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    assert(version != nullptr);
    *version = HORUS_CODE_BLOCK_VERSION;
    return Horus_code_block_success;
}

// -- Optional discovery function ------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    assert(discovery_info != nullptr);
    static const std::string static_discovery_info_description =
        "Erase the four least significant bits of video buffer bytes of input messages "
        "arriving at the Data 0 input pipe. Then send the result to the Data output pipe "
        "indicated by '" +
        _video_output_pipe_key_ +
        "'.\n"
        "\n"
        "This shared library demonstration is only useful on raw video.";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "LSB Eraser", static_discovery_info_description.c_str()
    };
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

// -- Mandatory functions --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const context,
    Horus_code_block_user_context **const user_context)
{
    assert((context != nullptr) && (user_context != nullptr));
    *user_context = new User_context(context->key_value_array);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    assert((context != nullptr) && (context->user_context != nullptr));
    delete static_cast<User_context *>(context->user_context);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    assert((context != nullptr) && (context->user_context != nullptr) && (data != nullptr));
    auto &user_context = *static_cast<User_context *>(context->user_context);
    return user_context.handle_write(*data);
}

Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    assert((context != nullptr) && (context->user_context != nullptr) && (data != nullptr));
    auto &user_context = *static_cast<User_context *>(context->user_context);
    return user_context.handle_read(*data);
}
