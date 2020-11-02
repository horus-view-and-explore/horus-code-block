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

#ifndef HORUS_CODE_BLOCK_OUTPUT_MESSAGE_BASE_HPP
#define HORUS_CODE_BLOCK_OUTPUT_MESSAGE_BASE_HPP

#include "../Horus_code_block.h"

class Horus_code_block_output_message_base
{
    // =============================================================================================
    // -- Special member functions -----------------------------------------------------------------
    // =============================================================================================

  protected:
    Horus_code_block_output_message_base(Horus_code_block_output_pipe output_pipe);

  public:
    Horus_code_block_output_message_base(const Horus_code_block_output_message_base &) = delete;

    Horus_code_block_output_message_base &
    operator=(const Horus_code_block_output_message_base &) = delete;

    Horus_code_block_output_message_base(Horus_code_block_output_message_base &&) = delete;

    Horus_code_block_output_message_base &
    operator=(Horus_code_block_output_message_base &&) = delete;

    virtual ~Horus_code_block_output_message_base() = default;

    // =============================================================================================
    // -- Other member functions -------------------------------------------------------------------
    // =============================================================================================

    bool data_output_message_begin_available() const;

    virtual bool data_available() const = 0;

    void set_data_output_message_begin();

    void ensure_data_output_message_begin();

    const struct Horus_code_block_data_output_message_begin &get_data_output_message_begin();

    // =============================================================================================
    // -- Member data ------------------------------------------------------------------------------
    // =============================================================================================

  private:
    bool _data_output_message_begin_available;

    struct Horus_code_block_data_output_message_begin _data_output_message_begin;
};

#endif // HORUS_CODE_BLOCK_OUTPUT_MESSAGE_BASE_HPP
