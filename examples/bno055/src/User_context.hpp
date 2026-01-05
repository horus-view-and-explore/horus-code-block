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

#ifndef USER_CONTEXT_HPP
#define USER_CONTEXT_HPP

#include "../../../Horus_code_block.h"
#include "BNO055.hpp"
#include "Gpio_trigger.hpp"

namespace horus::code::bno055 {
class User_context
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  public:
    User_context(const std::string device, const uint32_t pps_gpio);
    ~User_context();

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
        Send_data_buffered,
        Send_data_sequence,
        Send_data_timestamp,
        Do_nothing
    } _next_read_action;

    void pps_callback();

    std::string _i2c_device;

    uint32_t _pps_gpio;

    Horus_code_block_data_output_message_begin _out_msg_begin;

    Horus_code_block_unit _euler_unit;
    Horus_code_block_unit _sequence_unit;

    int32_t _euler_dimension;

    double _euler_value[3];

    int32_t _metadata_dimension;

    Horus_code_block_data_sensor _data_sensor{};

    Horus_code_block_data _data;

    BNO055 _imu;

    uint64_t _sequence_number;

    BNO055::Euler _buffered_euler;
    uint64_t _buffered_timestamp;

    Gpio_trigger _trigger;

    bool _buffered_data_available;

    double _buffered_euler_data[3];
    Horus_code_block_data_sensor _buffered_sensor{};
    Horus_code_block_data_sensor _sequence_sensor{};
    Horus_code_block_data_sensor _timestamp_sensor{};
};

} // namespace horus::code::bno055

#endif /* USER_CONTEXT_HPP */
