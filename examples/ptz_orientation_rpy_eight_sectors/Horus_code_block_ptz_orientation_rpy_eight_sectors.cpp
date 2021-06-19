// This file contains PTZ Orientation RPY Eight Sectors, a C++ example
// implementations of the Horus Code Block C API.
//
// Copyright (C) 2021 Horus View and Explore B.V.
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
#include <cmath>
#include <string>

namespace {

// =================================================================================================
// -- Internal data --------------------------------------------------------------------------------
// =================================================================================================

const std::string _offset_key_("offset");

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

    void set_offset(const double offset)
    {
        _offset = offset;
    }

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

    const Horus_code_block_data_output_message_begin _data_output_message_begin;

    // -- outgoing sensor data ---------------------------------------------------------------------

    const Horus_code_block_unit _data_sensor_unit;

    const int32_t _data_sensor_dimension;

    double _data_sensor_double_value_array[8];

    const int32_t _data_sensor_metadata_dimension;

    Horus_code_block_data_sensor _data_sensor;

    Horus_code_block_data _data;

    // -- internal data ----------------------------------------------------------------------------

    double _offset;
};

// =================================================================================================
// -- Special member functions ---------------------------------------------------------------------
// =================================================================================================

User_context::User_context()
    : _next_read_action(Next_read_action::Do_nothing)
    , _data_output_message_begin{Horus_code_block_output_pipe_data_0} // output_pipe
    , _data_sensor_unit(Horus_code_block_unit_unknown)
    , _data_sensor_dimension(8)
    , _data_sensor_double_value_array()
    , _data_sensor_metadata_dimension(0)
    , _data_sensor{"ptz orientation yaw array",         // name
                   &_data_sensor_unit,                  // unit
                   Horus_code_block_sensor_data_double, // data
                   1u,                                  // dimension_array_size
                   &_data_sensor_dimension,             // dimension_array
                   nullptr,                             // string_value_array
                   nullptr,                             // integer_value_array
                   _data_sensor_double_value_array,     // double_value_array
                   nullptr,                             // float_value_array
                   nullptr,                             // long_value_array
                   nullptr,                             // unsigned_integer_value_array
                   nullptr,                             // unsigned_long_value_array
                   1u,                                  // metadata_dimension_array_size
                   &_data_sensor_metadata_dimension,    // metadata_dimension_array
                   nullptr}                             // metadata_array
    , _data()
    , _offset(0.0)
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
                *static_cast<const struct Horus_code_block_data_input_message_begin *>(
                    data.contents);
            if (data_input_message_begin.input_pipe == Horus_code_block_input_pipe_clock)
            {
                _next_read_action = Next_read_action::Send_data_output_message_begin;
            }
            break;
        }
        case Horus_code_block_data_type_ptz_orientation_rpy:
        {
            const auto &horus_code_block_data_ptz_orientation_rpy =
                *static_cast<const struct Horus_code_block_data_ptz_orientation_rpy *>(
                    data.contents);

            double yaw = 0.0;
            switch (*horus_code_block_data_ptz_orientation_rpy.yaw_unit)
            {
                case Horus_code_block_unit_radian:
                {
                    static const double quarter_pi = std::atan(1.0);
                    auto to_degrees = [](const double radians) -> double {
                        return radians / quarter_pi * 45.0;
                    };
                    // Assume offset and input have the same units.
                    yaw = to_degrees(horus_code_block_data_ptz_orientation_rpy.yaw) +
                          to_degrees(_offset);
                    break;
                }
                case Horus_code_block_unit_degree:
                default:
                    // Assume the user forgot to set units and assume degrees.
                    yaw = horus_code_block_data_ptz_orientation_rpy.yaw + _offset;
            }

            // Map to [0, 360).
            auto normalize = [](double angle) -> double {
                angle = std::fmod(angle, 360.0);
                if (angle < 0.0)
                {
                    angle += 360.0;
                }
                return angle;
            };
            yaw = normalize(yaw);

            auto in_right_side_open_interval =
                [](const double a, const double b, const double x) -> bool {
                return (a <= x) && (x < b);
            };
            auto map_onto_interval =
                [](const double old_max, const double new_max, const double value) -> double {
                return new_max * value / old_max;
            };

            for (int32_t i = 0; i != _data_sensor_dimension; ++i)
            {
                const double left_side = static_cast<double>(i);
                const double right_side = left_side + 1.0;
                const double mapped = map_onto_interval(360, 8, yaw);
                const bool in_interval = in_right_side_open_interval(left_side, right_side, mapped);
                _data_sensor_double_value_array[i] = in_interval ? 1.0 : 0.0;
            }
            break;
        }
        default:
            return Horus_code_block_success;
    }
    return Horus_code_block_success;
}

Horus_code_block_result User_context::handle_read(const Horus_code_block_data *&data)
{
    switch (_next_read_action)
    {
        case Next_read_action::Do_nothing:
            return Horus_code_block_success;
        case Next_read_action::Send_data_output_message_begin:
            _data.type = Horus_code_block_data_type_output_message_begin;
            _data.contents = &_data_output_message_begin;
            data = &_data;
            _next_read_action = Next_read_action::Send_data_sensor;
            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        case Next_read_action::Send_data_sensor:
            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_data_sensor;
            data = &_data;
            _next_read_action = Next_read_action::Do_nothing;
            return Horus_code_block_success;
        default:
            assert(false);
            return Horus_code_block_error;
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
        "Split the unit circle into 8 parts and check where RPY values fall.\n"
        "\n"
        "Supported keys: '" +
        _offset_key_ + "'.";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "PTZ Orientation RPY Eight Sectors", static_discovery_info_description.c_str()
    };
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

// -- Mandatory functions --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const context,
    Horus_code_block_user_context **const user_context)
{
    *user_context = new User_context;
    for (size_t i = 0u; i != context->key_value_array.size; ++i)
    {
        const struct Horus_code_block_key_value key_value = context->key_value_array.array[i];
        if (key_value.key == _offset_key_)
        {
            static_cast<User_context *>(*user_context)->set_offset(std::stod(key_value.value));
        }
    }
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
