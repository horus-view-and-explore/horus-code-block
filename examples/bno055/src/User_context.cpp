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

#include "User_context.hpp"

#include <cstring>
#include <iostream>

namespace horus::code::bno055 {

User_context::User_context(const std::string device, const unsigned int pps_gpio)
    : _next_read_action(Next_read_action::Do_nothing)
    , _i2c_device(device)
    , _pps_gpio(pps_gpio)
    , _out_msg_begin{Horus_code_block_output_pipe_data_0}
    , _euler_unit(Horus_code_block_unit_radian)
    , _sequence_unit(Horus_code_block_unit_unknown)
    , _euler_dimension(3)
    , _euler_value{0.0, 0.0, 0.0}
    , _metadata_dimension(0)
    , _data()
    , _imu(device, BNO055::OutputFormat::FmtEuler)
    , _sequence_number(0)
    , _buffered_euler{}
    , _buffered_timestamp()
    , _trigger(pps_gpio, Gpio_trigger::GpioEdge::Rising, 0, [this]() { pps_callback(); })
    , _buffered_data_available(false)
{
    _data_sensor.name = "BNO055 Orientation";
    _data_sensor.unit = &_euler_unit;
    _data_sensor.data = Horus_code_block_sensor_data_double;
    _data_sensor.dimension_array_size = 1u;
    _data_sensor.dimension_array = &_euler_dimension;
    _data_sensor.double_value_array = _euler_value;
    _data_sensor.metadata_dimension_array_size = 1u;
    _data_sensor.metadata_dimension_array = &_metadata_dimension;

    _buffered_sensor.name = "BNO055 Orientation (Tagged)";
    _buffered_sensor.unit = &_euler_unit;
    _buffered_sensor.data = Horus_code_block_sensor_data_double;
    _buffered_sensor.dimension_array_size = 1u;
    _buffered_sensor.dimension_array = &_euler_dimension;
    _buffered_sensor.double_value_array = _buffered_euler_data;
    _buffered_sensor.metadata_dimension_array_size = 1u;
    _buffered_sensor.metadata_dimension_array = &_metadata_dimension;

    _sequence_sensor.name = "BNO055 Sequence";
    _sequence_sensor.unit = &_sequence_unit;
    _sequence_sensor.data = Horus_code_block_sensor_data_unsigned_long;
    _sequence_sensor.dimension_array_size = 0u;
    _sequence_sensor.unsigned_long_value_array = &_sequence_number;
    _sequence_sensor.metadata_dimension_array_size = 1u;
    _sequence_sensor.metadata_dimension_array = &_metadata_dimension;

    _timestamp_sensor.name = "BNO055 Timestamp";
    _timestamp_sensor.unit = &_sequence_unit;
    _timestamp_sensor.data = Horus_code_block_sensor_data_unsigned_long;
    _timestamp_sensor.dimension_array_size = 0u;
    _timestamp_sensor.unsigned_long_value_array = &_buffered_timestamp;
    _timestamp_sensor.metadata_dimension_array_size = 1u;
    _timestamp_sensor.metadata_dimension_array = &_metadata_dimension;

    _trigger.open();
    _imu.init();
    _imu.start();
}

User_context::~User_context()
{
    _imu.stop();
}

Horus_code_block_result User_context::handle_write(const struct Horus_code_block_data &data)
{
    if (data.type == Horus_code_block_data_type_input_message_begin)
    {
        const auto &data_input_message_begin =
            *reinterpret_cast<const struct Horus_code_block_data_input_message_begin *>(
                data.contents);
        if (data_input_message_begin.input_pipe == Horus_code_block_input_pipe_clock)
        {
            _next_read_action = Next_read_action::Send_data_output_message_begin;
        }
    }
    return Horus_code_block_success;
}

Horus_code_block_result User_context::handle_read(const struct Horus_code_block_data *&data)
{
    switch (_next_read_action)
    {
        case Next_read_action::Do_nothing:
            return Horus_code_block_success;
        case Next_read_action::Send_data_output_message_begin:
        {
            if (_buffered_data_available)
            {
                _next_read_action = Next_read_action::Send_data_buffered;
                _out_msg_begin.output_pipe = Horus_code_block_output_pipe_data_1;
            }
            else
            {
                _next_read_action = Next_read_action::Send_data_sensor;
                _out_msg_begin.output_pipe = Horus_code_block_output_pipe_data_0;
            }

            _data.type = Horus_code_block_data_type_output_message_begin;
            _data.contents = &_out_msg_begin;

            data = &_data;

            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sensor:
        {
            const BNO055::Euler euler = _imu.euler();

            _euler_value[0] = euler.x;
            _euler_value[1] = euler.y;
            _euler_value[2] = euler.z;

            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_data_sensor;
            data = &_data;
            _next_read_action = Next_read_action::Do_nothing;
            return Horus_code_block_success;
        }
        case Next_read_action::Send_data_buffered:
        {
            _buffered_euler_data[0] = _buffered_euler.x;
            _buffered_euler_data[1] = _buffered_euler.y;
            _buffered_euler_data[2] = _buffered_euler.z;

            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_buffered_sensor;
            data = &_data;
            _next_read_action = Next_read_action::Send_data_sequence;
            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_sequence:
        {
            _sequence_number++;
            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_sequence_sensor;
            data = &_data;
            _next_read_action = Next_read_action::Send_data_timestamp;
            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        case Next_read_action::Send_data_timestamp:
        {
            _data.type = Horus_code_block_data_type_sensor;
            _data.contents = &_timestamp_sensor;
            data = &_data;
            _next_read_action = Next_read_action::Send_data_output_message_begin;

            _buffered_data_available = false;

            return Horus_code_block_success | Horus_code_block_read_flag_read_more;
        }
        default:
        {
            return Horus_code_block_error;
        }
    }
}

void User_context::pps_callback()
{
    _buffered_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();
    _buffered_euler = _imu.euler();

    _buffered_data_available = true;
}
} // namespace horus::code::bno055
