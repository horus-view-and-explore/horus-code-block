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

#ifndef HORUS_CODE_GPIO_TRIGGER
#define HORUS_CODE_GPIO_TRIGGER

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <thread>

namespace horus::code::bno055 {

class Gpio_trigger
{
  public:
    enum GpioEdge
    {
        None,
        Rising,
        Falling,
        Both
    };

    enum GpioState
    {
        Unknown,
        High,
        Low
    };

    using Callback_type = std::function<void()>;

    Gpio_trigger(uint32_t pin, GpioEdge edge, uint32_t debounce_ms, Callback_type callback);
    ~Gpio_trigger();

    bool open();

    GpioState operator()();

  private:
    bool export_pin();
    void unexport_pin();
    bool set_mode();
    bool set_edge();

    bool register_interrupt();

    void run();

    const uint32_t _pin;
    GpioEdge _edge;

    const uint32_t _debounce_ms;

    std::fstream _value;
    bool _exported;

    Callback_type _callback;

    std::atomic_bool _running;
    std::thread _thread;
};

} // namespace horus::code::bno055

#endif /* GPIO_TRIGGER_HPP */
