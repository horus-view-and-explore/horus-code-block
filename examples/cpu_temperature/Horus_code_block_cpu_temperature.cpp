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

#include "../../Horus_code_block.h"

#include <cassert>
#include <fstream>
#include <string>

namespace {

// =================================================================================================
// -- Internal data --------------------------------------------------------------------------------
// =================================================================================================

const std::string _temperature_file_name_("/sys/class/thermal/thermal_zone0/temp");

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

    Horus_code_block_result handle_write(const struct Horus_code_block_data &data);

    Horus_code_block_result handle_read(const struct Horus_code_block_data *&data);

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    enum class Next_read_action
    {
        Send_data_output_message_begin,
        Send_data_sensor,
        Do_nothing
    } _next_read_action;

    Horus_code_block_data_output_message_begin _data_output_message_begin;

    Horus_code_block_unit _data_sensor_unit;

    double _data_sensor_double_value;

    int32_t _data_sensor_metadata_dimension;

    Horus_code_block_data_sensor _data_sensor;

    Horus_code_block_data _data;
};

// =================================================================================================
// -- Special member functions ---------------------------------------------------------------------
// =================================================================================================

User_context::User_context()
    : _next_read_action(Next_read_action::Do_nothing)
    , _data_output_message_begin{Horus_code_block_output_pipe_data_0} // output_pipe
    , _data_sensor_unit(Horus_code_block_unit_celcius)
    , _data_sensor_double_value(0.0)
    , _data_sensor_metadata_dimension(0)
    , _data_sensor{"CPU Temperature",                   // name
                   &_data_sensor_unit,                  // unit
                   Horus_code_block_sensor_data_double, // data
                   0u,                                  // dimension_array_size
                   nullptr,                             // dimension_array
                   nullptr,                             // string_value_array
                   nullptr,                             // integer_value_array
                   &_data_sensor_double_value,          // double_value_array
                   nullptr,                             // float_value_array
                   nullptr,                             // long_value_array
                   nullptr,                             // unsigned_integer_value_array
                   nullptr,                             // unsigned_long_value_array
                   1u,                                  // metadata_dimension_array_size
                   &_data_sensor_metadata_dimension,    // metadata_dimension_array
                   nullptr}                             // metadata_array
    , _data()
{
}

// =================================================================================================
// -- Other member functions -----------------------------------------------------------------------
// =================================================================================================

Horus_code_block_result User_context::handle_write(const Horus_code_block_data &data)
{
    if (data.type == Horus_code_block_data_type_input_message_begin)
    {
        assert(_next_read_action == Next_read_action::Do_nothing);
        const auto &data_input_message_begin =
            *static_cast<const struct Horus_code_block_data_input_message_begin *>(data.contents);
        if (data_input_message_begin.input_pipe == Horus_code_block_input_pipe_clock)
        {
            _next_read_action = Next_read_action::Send_data_output_message_begin;
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
            _data.type = Horus_code_block_data_type_output_message_begin;
            _data.contents = &_data_output_message_begin;
            data = &_data;
            _next_read_action = Next_read_action::Send_data_sensor;
            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sensor:
        {
            std::ifstream temperature_file(_temperature_file_name_);
            if (temperature_file)
            {
                temperature_file >> _data_sensor_double_value;
                _data_sensor_double_value /= 1000.0; // to degrees celcius
            }
            else
            {
                _data_sensor_double_value = 0.0;
            }
            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_data_sensor;
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
    static const std::string static_discovery_info_description =
        "Produce CPU temperature readings as sensor data.\n"
        "\n"
        "For each input message arriving at the Clock input pipe, produce an output message that "
        "contains the CPU temperature as sensor data and send it to the Data 0 output pipe.\n"
        "\n"
        "This components reads the temperature from the file '" +
        _temperature_file_name_ + "'.";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "CPU Temperature", static_discovery_info_description.c_str()
    };
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
    auto user_context = static_cast<User_context *>(context->user_context);
    return user_context->handle_read(*data);
}
