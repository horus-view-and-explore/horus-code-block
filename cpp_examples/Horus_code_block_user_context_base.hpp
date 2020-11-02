// This file is part of the C++ example implementations of the Horus Code Block C API.
//
// Copyright (C) 2020 Horus View and Explore B.V.
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

#ifndef HORUS_CODE_BLOCK_USER_CONTEXT_BASE_HPP
#define HORUS_CODE_BLOCK_USER_CONTEXT_BASE_HPP

#include "../Horus_code_block.h"

#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

class Horus_code_block_user_context_base
{
    // =============================================================================================
    // -- Member types -----------------------------------------------------------------------------
    // =============================================================================================

  private:
    struct Log
    {
        Horus_code_block_log_severity severity;

        std::string text;
    };

    struct Subscriptions
    {
        std::set<Horus_code_block_input_pipe> input_pipe_set;

        std::set<Horus_code_block_data_type> input_message_data_type_subscription_set;
    };

    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  protected:
    Horus_code_block_user_context_base();

  public:
    Horus_code_block_user_context_base(const Horus_code_block_user_context_base &) = delete;

    Horus_code_block_user_context_base &
    operator=(const Horus_code_block_user_context_base &) = delete;

    Horus_code_block_user_context_base(Horus_code_block_user_context_base &&) = delete;

    Horus_code_block_user_context_base &operator=(Horus_code_block_user_context_base &&) = delete;

    virtual ~Horus_code_block_user_context_base() = default;

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    Horus_code_block_result handle_read(const struct Horus_code_block_data *&data);

    // -- Data -------------------------------------------------------------------------------------

  protected:
    virtual bool data_available() const;

    virtual void get_first_available_data(const struct Horus_code_block_data *&data);

    struct Horus_code_block_data *
    get_data(const struct Horus_code_block_data_output_message_begin &data_output_message_begin);

    struct Horus_code_block_data *
    get_data(const struct Horus_code_block_data_ascii_out &data_ascii_out);

    struct Horus_code_block_data *
    get_data(const struct Horus_code_block_data_video_fourcc_out &data_video_fourcc_out);

    struct Horus_code_block_data *get_data(const struct Horus_code_block_data_sensor &data_sensor);

  private:
    struct Horus_code_block_data *get_data(const struct Horus_code_block_data_log &data_log);

    struct Horus_code_block_data *
    get_data(const struct Horus_code_block_data_subscriptions &data_subscriptions);

    // -- Log data ---------------------------------------------------------------------------------

  protected:
    void push_log(const Horus_code_block_log_severity severity, std::string text);

  private:
    struct Horus_code_block_data *pop_log();

    // -- Subscriptions data -----------------------------------------------------------------------

  protected:
    /// @pre @a input_pipe_set is not empty.
    void push_subscriptions(
        std::set<Horus_code_block_input_pipe> input_pipe_set,
        std::set<Horus_code_block_data_type> input_message_data_type_subscription_set);

  private:
    struct Horus_code_block_data *pop_subscriptions();

    // -- Key-value pairs --------------------------------------------------------------------------

  protected:
    std::unique_ptr<Horus_code_block_output_pipe> get_data_output_pipe(
        const struct Horus_code_block_key_value_array &key_value_array,
        const std::string &data_output_pipe_key);

    Horus_code_block_output_pipe get_data_output_pipe_or(
        const struct Horus_code_block_key_value_array &key_value_array,
        const std::string &data_output_pipe_key,
        Horus_code_block_output_pipe default_data_output_pipe);

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

    // -- Data -------------------------------------------------------------------------------------

  private:
    struct Horus_code_block_data _data;

    // -- Log data ---------------------------------------------------------------------------------

    std::queue<Log> _log_queue;

    std::string _data_log_text;

    struct Horus_code_block_data_log _data_log;

    // -- Subscriptions data -----------------------------------------------------------------------

    std::queue<Subscriptions> _subscriptions_queue;

    std::vector<Horus_code_block_input_pipe> _data_subscriptions_input_pipe_vector;

    std::vector<Horus_code_block_data_type>
        _data_subscriptions_input_message_data_type_subscription_vector;

    struct Horus_code_block_data_subscriptions _data_subscriptions;
};

#endif // HORUS_CODE_BLOCK_USER_CONTEXT_BASE_HPP
