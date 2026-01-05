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
#include <iostream>
#include <sstream>
#include <string>

#include "User_context.hpp"

inline std::string log_header(const std::string &component_id)
{
    std::stringstream ss;
    ss << "[BNO055 " << component_id << "] ";
    return ss.str();
}

// =================================================================================================
// -- API ------------------------------------------------------------------------------------------
// =================================================================================================

// -- Mandatory version function -------------------------------------------------------------------

[[maybe_unused]] Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    *version = HORUS_CODE_BLOCK_VERSION;
    return Horus_code_block_success;
}

[[maybe_unused]] Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    static const std::string static_discovery_info_description =
        "Produces a message on a PPS signal with the IMU data and GPS time.\n"
        "\n"
        "When an input message arrives at the Clock input pipe, the latest data is produced"
        "on the Data 0 output pipe if one is available. ";

    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "BNO055", static_discovery_info_description.c_str()
    };
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

[[maybe_unused]] Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const context,
    Horus_code_block_user_context **const user_context)
{
    std::cout << log_header(context->component_instance_id) << "Open" << std::endl;
    std::cout << log_header(context->component_instance_id)
              << "Key Value pairs: " << std::to_string(context->key_value_array.size) << std::endl;
    if (context->key_value_array.size > 0)
    {
        std::string i2c_dev = "/dev/i2c-1";
        unsigned int pps_gpio = 18;
        for (size_t i = 0u; i < context->key_value_array.size; i++)
        {
            const struct Horus_code_block_key_value key_value = context->key_value_array.array[i];
            std::cout << log_header(context->component_instance_id) << std::string(key_value.key)
                      << ": " << std::string(key_value.value) << std::endl;
            if (std::string(key_value.key) == "i2c-device")
            {
                i2c_dev = std::string(key_value.value);
            }
            else if (std::string(key_value.key) == "pps-gpio")
            {
                pps_gpio = std::strtoul(key_value.value, nullptr, 10);
            }
        }

        std::cout << log_header(context->component_instance_id) << "Using I2C device '" << i2c_dev
                  << "' and PPS GPIO pin '" << std::to_string(pps_gpio) << "'" << std::endl;

        bool valid = true;
        if (i2c_dev.empty())
        {
            std::cerr << "I2C device not specified, please specify the I2C device in the "
                         "'i2c-device' property"
                      << std::endl;
            valid = false;
        }
        if (pps_gpio == 0)
        {
            std::cerr << "Invalid PPS GPIO number, please specify a valid number in the 'pps-gpio' "
                         "property"
                      << std::endl;
            valid = false;
        }

        if (!valid)
        {
            return Horus_code_block_error;
        }

        *user_context = new horus::code::bno055::User_context(i2c_dev, pps_gpio);

        return Horus_code_block_success;
    }

    return Horus_code_block_error;
}

[[maybe_unused]] Horus_code_block_result
horus_code_block_close(const struct Horus_code_block_context *const context)
{
    std::cout << log_header(context->component_instance_id) << "Close" << std::endl;

    delete reinterpret_cast<horus::code::bno055::User_context *>(context->user_context);

    return Horus_code_block_success;
}

[[maybe_unused]] Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    auto user_context =
        reinterpret_cast<horus::code::bno055::User_context *>(context->user_context);

    return user_context->handle_write(*data);
}

[[maybe_unused]] Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    auto user_context =
        reinterpret_cast<horus::code::bno055::User_context *>(context->user_context);

    return user_context->handle_read(*data);
}
