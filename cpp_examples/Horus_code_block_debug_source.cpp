// This file is part of the C++ example implementations of the Horus Code Block C API.
//
// Copyright (C) 2020 Horus View and Explore B.V.
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

#include "../Horus_code_block.h"

#include "Horus_code_block_output_message_base.hpp"
#include "Horus_code_block_user_context_base.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

namespace {

// =================================================================================================
// -- Internal data --------------------------------------------------------------------------------
// =================================================================================================

const std::string _ascii_sensor_output_pipe_key_("ascii-sensor-output-pipe");

const std::string _sensor_output_pipe_key_("sensor-output-pipe");

// =================================================================================================
// -- Internal functions ---------------------------------------------------------------------------
// =================================================================================================

template <typename OUTPUT_MESSAGE>
std::unique_ptr<OUTPUT_MESSAGE>
create_output_message(const std::unique_ptr<const Horus_code_block_output_pipe> output_pipe)
{
    if (output_pipe)
    {
        return std::make_unique<OUTPUT_MESSAGE>(*output_pipe);
    }
    else
    {
        return nullptr;
    }
}

double get_sensor_value(
    const struct Horus_code_block_data_grabber_utc_timestamp &data_grabber_utc_timestamp)
{
    const double timestamp_in_seconds = data_grabber_utc_timestamp.stamp * 1e-6;
    return std::sin(timestamp_in_seconds);
}

// =================================================================================================
// -- Internal types -------------------------------------------------------------------------------
// =================================================================================================

class Ascii_sensor_output_message : public Horus_code_block_output_message_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    Ascii_sensor_output_message(const Horus_code_block_output_pipe output_pipe)
        : Horus_code_block_output_message_base(output_pipe)
        , _data_ascii_out_available(false)
        , _data_ascii_out{nullptr, // data_id
                          0u,      // buffer_request_id
                          0u}      // buffer_size
    {
    }

    // =============================================================================================
    // -- Horus_code_block_output_message_base -----------------------------------------------------
    // =============================================================================================

    virtual bool data_available() const final override
    {
        return (
            Horus_code_block_output_message_base::data_available() || data_ascii_out_available());
    }

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    bool data_ascii_out_available() const
    {
        return _data_ascii_out_available;
    }

    void set_data_ascii_out(
        const struct Horus_code_block_data_grabber_utc_timestamp &data_grabber_utc_timestamp,
        std::map<Horus_code_block_request_id, std::string> &buffer_request_id_to_buffer_map)
    {
        assert(data_output_message_begin_available() && !_data_ascii_out_available);

        _data_ascii_out_available = true;
        std::string data_ascii_out_buffer =
            "NAME CodeAsciiBlockSensor UNITS NORMALIZED STRUCTURE SCALAR DATATYPE DOUBLE VALUE " +
            std::to_string(get_sensor_value(data_grabber_utc_timestamp)) + '\n';
        _data_ascii_out.buffer_request_id = buffer_request_id_to_buffer_map.size();
        _data_ascii_out.buffer_size = data_ascii_out_buffer.size();

        buffer_request_id_to_buffer_map.emplace(
            _data_ascii_out.buffer_request_id, std::move(data_ascii_out_buffer));
    }

    const struct Horus_code_block_data_ascii_out &get_data_ascii_out()
    {
        assert(!data_output_message_begin_available() && _data_ascii_out_available);
        _data_ascii_out_available = false;
        return _data_ascii_out;
    }

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    bool _data_ascii_out_available;

    struct Horus_code_block_data_ascii_out _data_ascii_out;
};

class Sensor_output_message : public Horus_code_block_output_message_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    Sensor_output_message(const Horus_code_block_output_pipe output_pipe)
        : Horus_code_block_output_message_base(output_pipe)
        , _data_sensor_available(false)
        , _data_sensor_unit(Horus_code_block_unit_normalized)
        , _data_sensor_float_value(0.0f)
        , _data_sensor_metadata_dimension(0)
        , _data_sensor{"CodeBlockSensor",                  // name
                       &_data_sensor_unit,                 // unit
                       Horus_code_block_sensor_data_float, // data
                       0u,                                 // dimension_array_size
                       nullptr,                            // dimension_array
                       nullptr,                            // string_value_array
                       nullptr,                            // integer_value_array
                       nullptr,                            // double_value_array
                       &_data_sensor_float_value,          // float_value_array
                       nullptr,                            // long_value_array
                       nullptr,                            // unsigned_integer_value_array
                       nullptr,                            // unsigned_long_value_array
                       1u,                                 // metadata_dimension_array_size
                       &_data_sensor_metadata_dimension,   // metadata_dimension_array
                       nullptr}                            // metadata_array
    {
    }

    // =============================================================================================
    // -- Horus_code_block_output_message_base -----------------------------------------------------
    // =============================================================================================

    virtual bool data_available() const final override
    {
        return (Horus_code_block_output_message_base::data_available() || data_sensor_available());
    }

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    bool data_sensor_available() const
    {
        return _data_sensor_available;
    }

    void set_data_sensor(
        const struct Horus_code_block_data_grabber_utc_timestamp &data_grabber_utc_timestamp)
    {
        assert(data_output_message_begin_available() && !_data_sensor_available);
        _data_sensor_available = true;
        _data_sensor_float_value = get_sensor_value(data_grabber_utc_timestamp);
    }

    const struct Horus_code_block_data_sensor &get_data_sensor()
    {
        assert(!data_output_message_begin_available() && _data_sensor_available);
        _data_sensor_available = false;
        return _data_sensor;
    }

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    bool _data_sensor_available;

    const Horus_code_block_unit _data_sensor_unit;

    float _data_sensor_float_value;

    const int32_t _data_sensor_metadata_dimension;

    const struct Horus_code_block_data_sensor _data_sensor;
};

class User_context final : public Horus_code_block_user_context_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    User_context(const struct Horus_code_block_key_value_array &key_value_array)
        : Horus_code_block_user_context_base()
        , _ascii_sensor_output_message(create_output_message<Ascii_sensor_output_message>(
              get_data_output_pipe(key_value_array, _ascii_sensor_output_pipe_key_)))
        , _sensor_output_message(create_output_message<Sensor_output_message>(
              get_data_output_pipe(key_value_array, _sensor_output_pipe_key_)))
        , _buffer_request_id_to_buffer_map()
    {
        push_log(Horus_code_block_log_severity_info, "Opened.");
        /// Subscribe all data input pipes no none.
        push_subscriptions(
            {Horus_code_block_input_pipe_data_0,
             Horus_code_block_input_pipe_data_1,
             Horus_code_block_input_pipe_data_2},
            {});
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
            (_ascii_sensor_output_message && _ascii_sensor_output_message->data_available()) ||
            (_sensor_output_message && _sensor_output_message->data_available()));
    }

    virtual void get_first_available_data(const struct Horus_code_block_data *&data) final override
    {
        if (_ascii_sensor_output_message &&
            _ascii_sensor_output_message->data_output_message_begin_available())
        {
            data = get_data(_ascii_sensor_output_message->get_data_output_message_begin());
        }
        else if (
            _ascii_sensor_output_message &&
            _ascii_sensor_output_message->data_ascii_out_available())
        {
            data = get_data(_ascii_sensor_output_message->get_data_ascii_out());
        }
        else if (
            _sensor_output_message && _sensor_output_message->data_output_message_begin_available())
        {
            data = get_data(_sensor_output_message->get_data_output_message_begin());
        }
        else if (_sensor_output_message && _sensor_output_message->data_sensor_available())
        {
            data = get_data(_sensor_output_message->get_data_sensor());
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
                assert(
                    (!_ascii_sensor_output_message ||
                     !_ascii_sensor_output_message->data_available()) &&
                    (!_sensor_output_message || !_sensor_output_message->data_available()) &&
                    _buffer_request_id_to_buffer_map.empty());
                break;
            case Horus_code_block_data_type_grabber_utc_timestamp:
            {
                // First grabber UTC timestamp.
                assert(data.contents != nullptr);
                const auto &data_grabber_utc_timestamp =
                    *static_cast<const struct Horus_code_block_data_grabber_utc_timestamp *>(
                        data.contents);
                if (_ascii_sensor_output_message && !_ascii_sensor_output_message->data_available())
                {
                    _ascii_sensor_output_message->set_data_output_message_begin();
                    _ascii_sensor_output_message->set_data_ascii_out(
                        data_grabber_utc_timestamp, _buffer_request_id_to_buffer_map);
                }
                if (_sensor_output_message && !_sensor_output_message->data_available())
                {
                    _sensor_output_message->set_data_output_message_begin();
                    _sensor_output_message->set_data_sensor(data_grabber_utc_timestamp);
                }
                break;
            }
            case Horus_code_block_data_type_buffer:
                assert(data.contents != nullptr);
                write_buffer(
                    *static_cast<const struct Horus_code_block_data_buffer *>(data.contents));
                break;
            default:;
        }
        return Horus_code_block_success;
    }

  private:
    void write_buffer(const struct Horus_code_block_data_buffer &data_buffer)
    {
        const auto &buffer_request_id_and_buffer_pair =
            _buffer_request_id_to_buffer_map.find(data_buffer.buffer_request_id);
        assert(buffer_request_id_and_buffer_pair != _buffer_request_id_to_buffer_map.end());
        {
            const std::string &buffer = buffer_request_id_and_buffer_pair->second;
            assert(buffer.size() == data_buffer.buffer_size);
            std::memcpy(data_buffer.buffer, buffer.data(), buffer.size());
        }
        _buffer_request_id_to_buffer_map.erase(buffer_request_id_and_buffer_pair);
    }

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

    std::unique_ptr<Ascii_sensor_output_message> _ascii_sensor_output_message;

    std::unique_ptr<Sensor_output_message> _sensor_output_message;

    std::map<Horus_code_block_request_id, std::string> _buffer_request_id_to_buffer_map;
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
        "Produce various types of output messages.\n"
        "\n"
        "For each input message arriving at the Clock input pipe that contains a grabber UTC "
        "timestamp, produce a number of output messages. You can control which output messages "
        "Debug Sink produces and to which Data output pipes it sends them by specifying certain "
        "key-value pairs. The values are either '0', '1', '2' or '3' corresponding to a Data "
        "output pipe.\n"
        "\n"
        "'" +
        _ascii_sensor_output_pipe_key_ +
        "': Produce an output message with ASCII sensor data.\n"
        "'" +
        _sensor_output_pipe_key_ + "': Produce an output message with sensor data.\n";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "Debug Source", static_discovery_info_description.c_str()
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
