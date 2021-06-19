// This file is part of the C++ example implementations of the Horus Code Block C API.
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

#include "Horus_code_block_user_context_base.hpp"

#include <cassert>
#include <cstdlib>

namespace {

// =================================================================================================
// -- Internal functions ---------------------------------------------------------------------------
// =================================================================================================

// -- Key-value pairs ------------------------------------------------------------------------------

const char *
find_value(const struct Horus_code_block_key_value_array &key_value_array, const std::string &key)
{
    assert(key_value_array.array != nullptr);
    for (size_t i = 0u; i < key_value_array.size; ++i)
    {
        const struct Horus_code_block_key_value key_value = key_value_array.array[i];
        assert((key_value.key != nullptr) && (key_value.value != nullptr));
        if (key_value.key == key)
        {
            return key_value.value;
        }
    }
    return nullptr;
}

// -- Output pipes ---------------------------------------------------------------------------------

std::string get_output_pipe_text(const Horus_code_block_output_pipe output_pipe)
{
    switch (output_pipe)
    {
        case Horus_code_block_output_pipe_data_0:
            return "Data 0";
        case Horus_code_block_output_pipe_data_1:
            return "Data 1";
        case Horus_code_block_output_pipe_data_2:
            return "Data 2";
        case Horus_code_block_output_pipe_data_3:
            return "Data 3";
        default:
            assert(false);
            return std::string();
    }
}

} // namespace

// =================================================================================================
// -- Special member functions ---------------------------------------------------------------------
// =================================================================================================

Horus_code_block_user_context_base::Horus_code_block_user_context_base()
    : _data{}
    , _log_queue()
    , _data_log_text()
    , _data_log{}
    , _subscriptions_queue()
    , _data_subscriptions_input_pipe_vector()
    , _data_subscriptions_input_message_data_type_subscription_vector()
    , _data_subscriptions{}
{
}

// =================================================================================================
// -- Other member functions -----------------------------------------------------------------------
// =================================================================================================

Horus_code_block_result
Horus_code_block_user_context_base::handle_read(const struct Horus_code_block_data *&data)
{
    get_first_available_data(data);
    Horus_code_block_result result = Horus_code_block_success;
    if (data_available())
    {
        result |= Horus_code_block_read_flag_read_more;
    }
    return result;
}

// -- Data -----------------------------------------------------------------------------------------

bool Horus_code_block_user_context_base::data_available() const
{
    return (!_log_queue.empty() || !_subscriptions_queue.empty());
}

void Horus_code_block_user_context_base::get_first_available_data(
    const struct Horus_code_block_data *&data)
{
    if (!_log_queue.empty())
    {
        data = pop_log();
    }
    else if (!_subscriptions_queue.empty())
    {
        data = pop_subscriptions();
    }
}

struct Horus_code_block_data *Horus_code_block_user_context_base::get_data(
    const struct Horus_code_block_data_output_message_begin &data_output_message_begin)
{
    _data.type = Horus_code_block_data_type_output_message_begin;
    _data.contents = &data_output_message_begin;
    return &_data;
}

struct Horus_code_block_data *Horus_code_block_user_context_base::get_data(
    const struct Horus_code_block_data_ascii_out &data_ascii_out)
{
    _data.type = Horus_code_block_data_type_ascii_out;
    _data.contents = &data_ascii_out;
    return &_data;
}

struct Horus_code_block_data *Horus_code_block_user_context_base::get_data(
    const struct Horus_code_block_data_video_fourcc_out &data_video_fourcc_out)
{
    _data.type = Horus_code_block_data_type_video_fourcc_out;
    _data.contents = &data_video_fourcc_out;
    return &_data;
}

struct Horus_code_block_data *Horus_code_block_user_context_base::get_data(
    const struct Horus_code_block_data_ptz_orientation_rpy &data_ptz_orientation_rpy)
{
    _data.type = Horus_code_block_data_type_ptz_orientation_rpy;
    _data.contents = &data_ptz_orientation_rpy;
    return &_data;
}

struct Horus_code_block_data *
Horus_code_block_user_context_base::get_data(const struct Horus_code_block_data_sensor &data_sensor)
{
    _data.type = Horus_code_block_data_type_sensor;
    _data.contents = &data_sensor;
    return &_data;
}

struct Horus_code_block_data *
Horus_code_block_user_context_base::get_data(const struct Horus_code_block_data_log &data_log)
{
    _data.type = Horus_code_block_data_type_log;
    _data.contents = &data_log;
    return &_data;
}

struct Horus_code_block_data *Horus_code_block_user_context_base::get_data(
    const struct Horus_code_block_data_subscriptions &data_subscriptions)
{
    _data.type = Horus_code_block_data_type_subscriptions;
    _data.contents = &data_subscriptions;
    return &_data;
}

// -- Log data -------------------------------------------------------------------------------------

void Horus_code_block_user_context_base::push_log(
    const Horus_code_block_log_severity severity,
    std::string text)
{
    _log_queue.push(Log{severity, std::move(text)});
}

struct Horus_code_block_data *Horus_code_block_user_context_base::pop_log()
{
    assert(!_log_queue.empty());
    {
        const Log &log = _log_queue.front();
        _data_log.severity = log.severity;
        _data_log_text = log.text;
        _data_log.text = _data_log_text.c_str();
    }
    _log_queue.pop();
    return get_data(_data_log);
}

// -- Input message data type subscription data ----------------------------------------------------

void Horus_code_block_user_context_base::push_subscriptions(
    std::set<Horus_code_block_input_pipe> input_pipe_set,
    std::set<Horus_code_block_data_type> input_message_data_type_subscription_set)
{
    assert(!input_pipe_set.empty());
    _subscriptions_queue.push(Subscriptions{
        std::move(input_pipe_set), std::move(input_message_data_type_subscription_set)});
}

struct Horus_code_block_data *Horus_code_block_user_context_base::pop_subscriptions()
{
    assert(!_subscriptions_queue.empty());
    {
        const Subscriptions &subscriptions = _subscriptions_queue.front();
        _data_subscriptions_input_pipe_vector.assign(
            subscriptions.input_pipe_set.begin(), subscriptions.input_pipe_set.end());
        _data_subscriptions_input_message_data_type_subscription_vector.assign(
            subscriptions.input_message_data_type_subscription_set.begin(),
            subscriptions.input_message_data_type_subscription_set.end());
        _data_subscriptions.input_pipe_array_size = _data_subscriptions_input_pipe_vector.size();
        _data_subscriptions.input_pipe_array = _data_subscriptions_input_pipe_vector.data();
        _data_subscriptions.input_message_data_type_subscription_array_size =
            _data_subscriptions_input_message_data_type_subscription_vector.size();
        _data_subscriptions.input_message_data_type_subscription_array =
            _data_subscriptions_input_message_data_type_subscription_vector.data();
    }
    _subscriptions_queue.pop();
    return get_data(_data_subscriptions);
}

// -- Key-value pairs ------------------------------------------------------------------------------

std::unique_ptr<Horus_code_block_output_pipe>
Horus_code_block_user_context_base::get_data_output_pipe(
    const struct Horus_code_block_key_value_array &key_value_array,
    const std::string &data_output_pipe_key)
{
    const char *const data_output_pipe_value = find_value(key_value_array, data_output_pipe_key);
    if (data_output_pipe_value != nullptr)
    {
        const unsigned long int data_output_pipe =
            std::strtoul(data_output_pipe_value, nullptr, 10);
        switch (data_output_pipe)
        {
            case 0ul:
                return std::make_unique<Horus_code_block_output_pipe>(
                    Horus_code_block_output_pipe_data_0);
            case 1ul:
                return std::make_unique<Horus_code_block_output_pipe>(
                    Horus_code_block_output_pipe_data_1);
            case 2ul:
                return std::make_unique<Horus_code_block_output_pipe>(
                    Horus_code_block_output_pipe_data_2);
            case 3ul:
                return std::make_unique<Horus_code_block_output_pipe>(
                    Horus_code_block_output_pipe_data_3);
            default:
                push_log(
                    Horus_code_block_log_severity_warning,
                    "The value of key '" + data_output_pipe_key + "' is invalid.");
                return nullptr;
        }
    }
    else
    {
        return nullptr;
    }
}

Horus_code_block_output_pipe Horus_code_block_user_context_base::get_data_output_pipe_or(
    const struct Horus_code_block_key_value_array &key_value_array,
    const std::string &data_output_pipe_key,
    const Horus_code_block_output_pipe default_data_output_pipe)
{
    const std::unique_ptr<Horus_code_block_output_pipe> data_output_pipe =
        get_data_output_pipe(key_value_array, data_output_pipe_key);
    if (data_output_pipe)
    {
        return *data_output_pipe;
    }
    else
    {
        push_log(
            Horus_code_block_log_severity_warning,
            "No '" + data_output_pipe_key +
                "' key found or its value is invalid, using output pipe '" +
                get_output_pipe_text(default_data_output_pipe) + "' instead.");
        return default_data_output_pipe;
    }
}
