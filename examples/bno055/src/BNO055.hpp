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

#ifndef HORUS_CODE_BERRYGPS_BNO055
#define HORUS_CODE_BERRYGPS_BNO055

#include <cmath>
#include <cstdint>
#include <i2c/i2c.h>
#include <mutex>
#include <string>
#include <thread>

namespace horus::code::bno055 {

class BNO055
{
  private:
    static constexpr uint8_t imu_address = 0x29;
    // See BNO055 datasheet page 37, section 3.6.5.5, table 3-31: Quaternion data representation
    static constexpr double quaternion_scale = 1.0 / (1 << 14);
    static constexpr double euler_scale = 1.0 / 900;

  public:
    struct Quaternion
    {
        double w;
        double x;
        double y;
        double z;
    };

    struct Euler
    {
        double x;
        double y;
        double z;
    };

    enum OutputFormat
    {
        FmtQuat,
        FmtEuler
    };

    explicit BNO055(const std::string &i2c_bus_imu, OutputFormat out_fmt);
    ~BNO055();

    void init();
    void start();
    void stop();

    Quaternion quaternion();
    Euler euler();

  private:
    void read();
    void run();

    void readQuaternion();
    void readEuler();

    I2CDevice _imu_device{};

    Quaternion _quat{};
    Euler _euler{};

    std::thread _read_thread;
    std::mutex _mutex;
    bool _running;
    bool _initialized;
    OutputFormat _output_fmt;
};

} // namespace horus::code::bno055

#endif // HORUS_CODE_BERRYGPS_BNO055
