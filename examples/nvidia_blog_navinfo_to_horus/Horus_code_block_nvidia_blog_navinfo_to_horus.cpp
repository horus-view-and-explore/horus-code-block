// This file contains NavInfo JSON to Horus, a C++ implementation of the Horus
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../../Horus_code_block.h"
#include "lib/json.hpp"

#include <algorithm>
#include <cassert>
#include <exception>
#include <string>

namespace {

// =================================================================================================
// -- Internal types -------------------------------------------------------------------------------
// =================================================================================================

class User_context
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    User_context();

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    Horus_code_block_result handle_write(const Horus_code_block_data &data);

    Horus_code_block_result handle_read(const Horus_code_block_data *&data);

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

    // -- Write related data -----------------------------------------------------------------------

  private:
    Horus_code_block_input_pipe _input_pipe;

    nlohmann::json _input_json;

    // -- Read releated data -----------------------------------------------------------------------

    enum class Next_read_action
    {
        Do_nothing,
        Send_data_output_message_begin,
        Send_data_sensor_bbox,
        Send_data_sensor_category,
        Send_data_sensor_score,
        Send_data_log_error
    } _next_read_action;

    Horus_code_block_data_output_message_begin _data_output_message_begin;

    // -- Sensor output message data --

    // All three sensors have the same unit: unknown.
    Horus_code_block_unit _data_sensor_unit;

    // For bbox (vector).
    int32_t _data_sensor_dimension;

    // For category (1 string).
    std::string _data_sensor_string_value;
    const char *_data_sensor_string_value_array[1];

    // For bbox (4 doubles) and score (1 double).
    double _data_sensor_double_value_array[4];

    int32_t _data_sensor_metadata_dimension;

    Horus_code_block_data_sensor _data_sensor;

    // -- Log data --

    std::string _data_log_text;

    Horus_code_block_data_log _data_log;

    // -- Read data --

    Horus_code_block_data _data;
};

// =================================================================================================
// -- Special member functions ---------------------------------------------------------------------
// =================================================================================================

User_context::User_context()
    : _input_pipe()
    , _input_json()
    , _next_read_action(Next_read_action::Do_nothing)
    , _data_output_message_begin{Horus_code_block_output_pipe_data_0} // output pipe
    , _data_sensor_unit(Horus_code_block_unit_unknown)
    , _data_sensor_dimension(4)
    , _data_sensor_string_value()
    , _data_sensor_string_value_array()
    , _data_sensor_double_value_array()
    , _data_sensor_metadata_dimension(0)
    , _data_sensor{nullptr,                          // name, set later
                   &_data_sensor_unit,               // unit
                   0,                                // data, set later
                   0u,                               // dimension_array_size, set later
                   nullptr,                          // dimension_array, set later
                   nullptr,                          // string_value_array, set later
                   nullptr,                          // integer_value_array
                   nullptr,                          // double_value_array, set later
                   nullptr,                          // float_value_array
                   nullptr,                          // long_value_array
                   nullptr,                          // unsigned_integer_value_array
                   nullptr,                          // unsigned_long_value_array
                   1u,                               // metadata_dimension_array_size
                   &_data_sensor_metadata_dimension, // metadata_dimension_array
                   nullptr}                          // metadata_array
    , _data_log_text()
    , _data_log()
    , _data()
{
}

// =================================================================================================
// -- Other member functions -----------------------------------------------------------------------
// =================================================================================================

Horus_code_block_result User_context::handle_write(const Horus_code_block_data &data)
{
    switch (data.type)
    {
        case Horus_code_block_data_type_input_message_begin:
        {
            assert(_next_read_action == Next_read_action::Do_nothing);
            const auto &data_input_message_begin =
                *static_cast<const Horus_code_block_data_input_message_begin *>(data.contents);
            _input_pipe = data_input_message_begin.input_pipe;
            _input_json = nlohmann::json();
            break;
        }
        case Horus_code_block_data_type_ascii_in:
        {
            if (_input_pipe == Horus_code_block_input_pipe_data_0)
            {
                const auto &data_ascii_in =
                    *static_cast<const Horus_code_block_data_ascii_in *>(data.contents);
                std::string input_json_string(data_ascii_in.buffer, data_ascii_in.buffer_size);
                std::replace(input_json_string.begin(), input_json_string.end(), '\'', '\"');
                try
                {
                    _input_json = nlohmann::json::parse(std::move(input_json_string));
                }
                catch (const std::exception &)
                {
                    _data_log_text = "Failed to parse JSON from string: " + input_json_string;
                    _next_read_action = Next_read_action::Send_data_log_error;
                    return Horus_code_block_success;
                }
                if (!_input_json["objects"].empty())
                {
                    _next_read_action = Next_read_action::Send_data_output_message_begin;
                }
            }
            break;
        }
        default:
        {
            // Ignore.
        }
    }
    return Horus_code_block_success;
}

Horus_code_block_result User_context::handle_read(const Horus_code_block_data *&data)
{
    switch (_next_read_action)
    {
        case Next_read_action::Do_nothing:
        {
            return Horus_code_block_success;
        }
        case Next_read_action::Send_data_output_message_begin:
        {
            assert(!_input_json["objects"].empty());

            _data.type = Horus_code_block_data_type_output_message_begin;
            _data.contents = &_data_output_message_begin;
            data = &_data;

            _next_read_action = Next_read_action::Send_data_sensor_bbox;

            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sensor_bbox:
        {
            assert(!_input_json["objects"].empty());

            _data.type = Horus_code_block_data_type_sensor;
            static const std::string data_sensor_name("bbox");
            _data_sensor.name = data_sensor_name.c_str();
            _data_sensor.data = Horus_code_block_sensor_data_double;
            _data_sensor.dimension_array_size = 1u; // vector
            _data_sensor.dimension_array = &_data_sensor_dimension;
            _data_sensor.string_value_array = nullptr;
            try
            {
                // Parse from JSON.
                _data_sensor_double_value_array[0] = _input_json["objects"].at(0)["bbox"]["xmin"];
                _data_sensor_double_value_array[1] = _input_json["objects"].at(0)["bbox"]["ymin"];
                _data_sensor_double_value_array[2] = _input_json["objects"].at(0)["bbox"]["xmax"];
                _data_sensor_double_value_array[3] = _input_json["objects"].at(0)["bbox"]["ymax"];
            }
            catch (const std::exception &)
            {
                _data_log_text = "Failed to parse bbox from JSON.";
                _next_read_action = Next_read_action::Send_data_log_error;
                return Horus_code_block_success | Horus_code_block_read_flag_read_more;
            }
            _data_sensor.double_value_array = _data_sensor_double_value_array;
            _data.contents = &_data_sensor;
            data = &_data;

            _next_read_action = Next_read_action::Send_data_sensor_category;

            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sensor_category:
        {
            assert(!_input_json["objects"].empty());

            _data.type = Horus_code_block_data_type_sensor;
            static const std::string data_sensor_name("category");
            _data_sensor.name = data_sensor_name.c_str();
            _data_sensor.data = Horus_code_block_sensor_data_string;
            _data_sensor.dimension_array_size = 0u; // scalar
            _data_sensor.dimension_array = nullptr;
            try
            {
                // Parse from JSON.
                _data_sensor_string_value = _input_json["objects"].at(0)["category"];
            }
            catch (const std::exception &)
            {
                _data_log_text = "Failed to parse category from JSON.";
                _next_read_action = Next_read_action::Send_data_log_error;
                return Horus_code_block_success | Horus_code_block_read_flag_read_more;
            }
            _data_sensor_string_value_array[0] = _data_sensor_string_value.c_str();
            _data_sensor.string_value_array = _data_sensor_string_value_array;
            _data_sensor.double_value_array = nullptr;
            _data.contents = &_data_sensor;
            data = &_data;

            _next_read_action = Next_read_action::Send_data_sensor_score;

            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sensor_score:
        {
            assert(!_input_json["objects"].empty());

            _data.type = Horus_code_block_data_type_sensor;
            static const std::string data_sensor_name("score");
            _data_sensor.name = data_sensor_name.c_str();
            _data_sensor.data = Horus_code_block_sensor_data_double;
            _data_sensor.dimension_array_size = 0u; // scalar
            _data_sensor.dimension_array = nullptr;
            _data_sensor.string_value_array = nullptr;
            try
            {
                // Parse from JSON.
                _data_sensor_double_value_array[0] = _input_json["objects"].at(0)["score"];
            }
            catch (const std::exception &)
            {
                _data_log_text = "Failed to parse score from JSON.";
                _next_read_action = Next_read_action::Send_data_log_error;
                return Horus_code_block_success | Horus_code_block_read_flag_read_more;
            }
            _data_sensor.double_value_array = _data_sensor_double_value_array;
            _data.contents = &_data_sensor;
            data = &_data;

            // Completed this detection so erase it.
            _input_json["objects"].erase(0u);

            // Are there any more detections?
            if (!_input_json["objects"].empty())
            {
                _next_read_action = Next_read_action::Send_data_output_message_begin;
                return Horus_code_block_success | Horus_code_block_read_flag_read_more;
            }
            else
            {
                _next_read_action = Next_read_action::Do_nothing;
                return Horus_code_block_success;
            }
        }
        case Next_read_action::Send_data_log_error:
        {
            _data.type = Horus_code_block_data_type_log;
            _data_log.severity = Horus_code_block_log_severity_error;
            _data_log.text = _data_log_text.c_str();
            _data.contents = &_data_log;
            data = &_data;

            _next_read_action = Next_read_action::Do_nothing;

            return Horus_code_block_success;
        }
        default:
        {
            assert(false);
            return Horus_code_block_error;
        }
    }
}

} // namespace

// =================================================================================================
// -- API ------------------------------------------------------------------------------------------
// =================================================================================================

// -- Mandatory version function -------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    *version = HORUS_CODE_BLOCK_VERSION;
    return Horus_code_block_success;
}

// -- Optional discovery function ------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    static const std::string static_discovery_info_description(
        "Transform NavInfo detection JSON data to Horus sensor messages.\n"
        "\n"
        "For each input message that contains NavInfo JSON ASCII data that arrives at input pipe "
        "'Data 0', produce one output message for each detection in the JSON data and send it to "
        "the 'Data 0' output pipe. Each output message contains sensor data describing the "
        "detection.");
    static const Horus_code_block_discovery_info static_discovery_info{
        "AI JSON Parser", static_discovery_info_description.c_str()};
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

// -- Mandatory functions --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const,
    Horus_code_block_user_context **const user_context)
{
    *user_context = new User_context;
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    delete static_cast<User_context *>(context->user_context);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    auto &user_context = *static_cast<User_context *>(context->user_context);
    return user_context.handle_write(*data);
}

Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    auto *user_context = static_cast<User_context *>(context->user_context);
    return user_context->handle_read(*data);
}
