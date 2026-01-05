// BNO055 interfaces with the Bosch SensorTec BNO055 IMU
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

#include "BNO055.hpp"
#include "BNO055_def.hpp"

#include <iostream>
#include <string>

namespace horus::code::bno055 {

BNO055::BNO055(const std::string &i2c_bus_imu, OutputFormat out_fmt)
    : _read_thread()
    , _mutex()
    , _running(false)
    , _initialized(false)
    , _output_fmt(out_fmt)
{
    auto imu_bus = i2c_open(i2c_bus_imu.c_str());
    // auto compass_bus = i2c_open(i2c_bus_compass.c_str());

    i2c_init_device(&_imu_device);
    _imu_device.bus = imu_bus;
    _imu_device.addr = BNO055::imu_address;
}

BNO055::~BNO055()
{
    stop();
    i2c_close(_imu_device.bus);
    // i2c_close(_compass_device.bus);
}

void BNO055::init()
{
    using namespace std::chrono_literals;
    uint8_t data[1];

    // Reset chip
    data[0] = 0b00100000;
    i2c_ioctl_write(&_imu_device, BNO055_SYS_TRIGGER_ADDR, data, 1);

    // Wait until it comes back online
    std::this_thread::sleep_for(30ms);
    data[0] = 0;
    i2c_ioctl_read(&_imu_device, BNO055_CHIP_ID_ADDR, data, 1);
    while (data[0] != BNO055_ID)
    {
        std::this_thread::sleep_for(100ms);
        i2c_ioctl_read(&_imu_device, BNO055_CHIP_ID_ADDR, data, 1);
    }

    std::this_thread::sleep_for(50ms);

    // Set power mode to FULL POWER (Datasheet calls it 'normal')
    data[0] = 0;
    i2c_ioctl_write(&_imu_device, BNO055_PWR_MODE_ADDR, data, 1);
    std::this_thread::sleep_for(10ms);

    // Use external oscillator
    data[0] = 0b00000000;
    i2c_ioctl_write(&_imu_device, BNO055_SYS_TRIGGER_ADDR, data, 1);
    std::this_thread::sleep_for(10ms);

    // Set units to radians
    data[0] = 0b00000110;
    i2c_ioctl_write(&_imu_device, BNO055_UNIT_SEL_ADDR, data, 1);
    std::this_thread::sleep_for(10ms);

    // Set operation mode to NDOF
    data[0] = BNO055_OPERATION_MODE_NDOF;
    i2c_ioctl_write(&_imu_device, BNO055_OPR_MODE_ADDR, data, 1);
    std::this_thread::sleep_for(600ms); // BNO055_MODE_SWITCHING_DELAY

    _initialized = true;
}

BNO055::Quaternion BNO055::quaternion()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _quat;
}

BNO055::Euler BNO055::euler()
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _euler;
}

void BNO055::start()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_initialized)
    {
        init();
    }
    _running = true;
    std::thread t = std::thread([&]() { run(); });
    std::swap(t, _read_thread);
}

void BNO055::stop()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _running = false;
    if (_read_thread.joinable())
    {
        _read_thread.join();
    }
}

void BNO055::read()
{
    if (!_initialized)
    {
        return;
    }

    switch (_output_fmt)
    {
        case OutputFormat::FmtQuat:
            readQuaternion();
            break;
        case OutputFormat::FmtEuler:
            readEuler();
            break;
        default:
            std::cout << "Unknown output format " << _output_fmt << std::endl;
            break;
    }
}

void BNO055::readQuaternion()
{
    uint8_t buf[8];
    i2c_ioctl_read(&_imu_device, BNO055_QUATERNION_DATA_W_LSB_ADDR, buf, 8); // Read 8 bytes

    auto quat_w = static_cast<int16_t>(buf[0] | (buf[1] << 8));
    auto quat_x = static_cast<int16_t>(buf[2] | (buf[3] << 8));
    auto quat_y = static_cast<int16_t>(buf[4] | (buf[5] << 8));
    auto quat_z = static_cast<int16_t>(buf[6] | (buf[7] << 8));

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _quat = {
            BNO055::quaternion_scale * static_cast<double>(quat_w),
            BNO055::quaternion_scale * static_cast<double>(quat_x),
            BNO055::quaternion_scale * static_cast<double>(quat_y),
            BNO055::quaternion_scale * static_cast<double>(quat_z)};
    }
}

void BNO055::readEuler()
{
    uint8_t buf[6];
    i2c_ioctl_read(&_imu_device, BNO055_EULER_H_LSB_ADDR, buf, 6); // Read 6 bytes

    auto euler_x = static_cast<int16_t>(buf[0] | (buf[1] << 8));
    auto euler_y = static_cast<int16_t>(buf[2] | (buf[3] << 8));
    auto euler_z = static_cast<int16_t>(buf[4] | (buf[5] << 8));

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _euler = {
            BNO055::euler_scale * static_cast<double>(euler_x),
            BNO055::euler_scale * static_cast<double>(euler_y),
            BNO055::euler_scale * static_cast<double>(euler_z)};
    }
}

void BNO055::run()
{
    using namespace std::chrono_literals;

    while (_running)
    {
        auto start = std::chrono::high_resolution_clock::now();
        read();

        std::this_thread::sleep_until(start + 10ms);
    }
}
} // namespace horus::code::bno055
