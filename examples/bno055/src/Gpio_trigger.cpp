// Gpio_trigger calls a callback when an edge is detected on a GPIO
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

#include "Gpio_trigger.hpp"

#include <fcntl.h>
#include <iostream>
#include <sys/epoll.h>
#include <unistd.h>

namespace horus::code::bno055 {

namespace {
const std::string _sysfs_path = "/sys/class/gpio/";
const std::string _sysfs_export_path = _sysfs_path + "export";
const std::string _sysfs_unexport_path = _sysfs_path + "unexport";
} // namespace

Gpio_trigger::Gpio_trigger(
    const uint32_t pin,
    GpioEdge edge,
    const uint32_t debounce_ms,
    const Callback_type callback)
    : _pin(pin)
    , _edge(edge)
    , _debounce_ms(debounce_ms)
    , _value()
    , _exported(false)
    , _callback(callback)
    , _running(false)
    , _thread()
{
}

Gpio_trigger::~Gpio_trigger()
{
    _running.store(false);

    if (_thread.joinable())
    {
        _thread.join();
    }

    if (_exported)
    {
        unexport_pin();
    }
}

bool Gpio_trigger::open()
{
    return export_pin() && set_mode();
}

Gpio_trigger::GpioState Gpio_trigger::operator()()
{
    if (!_exported)
    {
        return GpioState::Unknown;
    }

    const std::string filename = _sysfs_path + "gpio" + std::to_string(_pin) + "/value";

    static std::fstream fs;
    fs.open(filename);

    if (!fs.is_open())
    {
        std::cerr << "Could not open GPIO" << std::endl;

        return GpioState::Unknown;
    }

    std::string data;
    fs >> data;

    if (fs.rdstate() & std::fstream::failbit)
    {
        std::cerr << "Unable to read GPIO state" << std::endl;
        fs.close();

        return GpioState::Unknown;
    }

    fs.close();

    if (data == "1")
    {
        return GpioState::High;
    }
    else if (data == "0")
    {
        return GpioState::Low;
    }

    return GpioState::Unknown;
}

bool Gpio_trigger::export_pin()
{
    std::fstream fs;
    fs.open(_sysfs_export_path, std::fstream::out);

    if (!fs.is_open())
    {
        std::cerr << "Could not export pin " << std::to_string(_pin)
                  << ", unable to open export file" << std::endl;
        return false;
    }

    fs << std::to_string(_pin) << std::endl;

    if (fs.rdstate() & std::fstream::failbit)
    {
        std::cerr << "Could not export pin " << std::to_string(_pin) << std::endl;

        fs.close();
        return false;
    }

    _exported = true;
    fs.close();

    std::cout << "GPIO " << std::to_string(_pin) << " exported" << std::endl;
    return true;
}

void Gpio_trigger::unexport_pin()
{
    if (!_exported)
    {
        return;
    }

    std::fstream fs;
    fs.open(_sysfs_unexport_path, std::fstream::out);

    if (!fs.is_open())
    {
        std::cerr << "Could not unexport pin " << std::to_string(_pin)
                  << ", unable to open unexport file" << std::endl;

        return;
    }

    fs << std::to_string(_pin) << std::endl;

    if (fs.rdstate() & std::fstream::failbit)
    {
        std::cerr << "Could not unexport pin " << std::to_string(_pin) << std::endl;
    }

    fs.close();
    std::cout << "GPIO " << std::to_string(_pin) << " unexported" << std::endl;
}

bool Gpio_trigger::set_mode()
{
    if (!_exported)
    {
        return false;
    }

    if (_value.is_open())
    {
        _value.close();
    }

    const std::string filename = _sysfs_path + "gpio" + std::to_string(_pin) + "/direction";

    std::fstream fs;

    fs.open(filename, std::fstream::out);

    if (!fs.is_open())
    {
        std::cerr << "Could not set pin " << std::to_string(_pin)
                  << " to input, could not open file" << std::endl;
    }

    fs << "in" << std::endl;

    if (fs.rdstate() & std::fstream::failbit)
    {
        std::cerr << "Could not set pin " << std::to_string(_pin) << " to input" << std::endl;
        fs.close();
        return false;
    }

    fs.close();

    std::cout << "GPIO " << std::to_string(_pin) << " mode set" << std::endl;
    return register_interrupt();
}

bool Gpio_trigger::set_edge()
{
    const std::string filename = _sysfs_path + "gpio" + std::to_string(_pin) + "/edge";

    std::fstream fs;
    fs.open(filename, std::fstream::out);

    if (!fs.good())
    {
        std::cerr << "Pin " << std::to_string(_pin) << " does not support interrupts" << std::endl;
        return false;
    }

    switch (_edge)
    {
        case GpioEdge::None:
            fs << "none" << std::endl;
            break;
        case GpioEdge::Rising:
            fs << "rising" << std::endl;
            break;
        case GpioEdge::Falling:
            fs << "falling" << std::endl;
            break;
        case GpioEdge::Both:
            fs << "both" << std::endl;
            break;
        default:
            break;
    }

    if (fs.rdstate() & std::ifstream::failbit)
    {
        std::cerr << "Could not set edge direction on pin " << std::to_string(_pin) << std::endl;
        fs.close();
        return false;
    }

    fs.close();

    std::cout << "GPIO " << std::to_string(_pin) << " edge set" << std::endl;
    return true;
}

bool Gpio_trigger::register_interrupt()
{
    if (!set_edge())
    {
        return false;
    }

    if (_edge == GpioEdge::None)
    {
        return true;
    }

    _running.store(true);

    std::thread t([this]() { run(); });
    std::swap(_thread, t);

    std::cout << "GPIO " << std::to_string(_pin) << " interrupt registered" << std::endl;
    return true;
}

void Gpio_trigger::run()
{
    const std::string value_filename = _sysfs_path + "gpio" + std::to_string(_pin) + "/value";
    int epfd = epoll_create(1);
    int gpiofd = ::open(value_filename.c_str(), O_RDONLY | O_NONBLOCK);

    if (gpiofd > 0)
    {
        char buf = 0;

        struct epoll_event ev;
        struct epoll_event events;
        ev.events = EPOLLPRI;
        ev.data.fd = gpiofd;

        int n = epoll_ctl(epfd, EPOLL_CTL_ADD, gpiofd, &ev);
        if (n < 0)
        {
            std::cerr << "Error adding GPIO file descriptor to epoll, aborting" << std::endl;
            return;
        }

        while (_running.load())
        {
            n = epoll_wait(epfd, &events, 1, 10000);

            if (n > 0)
            {
                lseek(gpiofd, 0, SEEK_SET);
                n = read(gpiofd, &buf, 1);
                if (n == 1 && buf == '1')
                {
                    _callback();
                }
            }
        }

        ::close(gpiofd);
    }
}
} // namespace horus::code::bno055
