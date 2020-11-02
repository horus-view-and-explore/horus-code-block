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

#include "Horus_code_block_output_message_base.hpp"

#include <cassert>

// =================================================================================================
// -- Special member functions ---------------------------------------------------------------------
// =================================================================================================

Horus_code_block_output_message_base::Horus_code_block_output_message_base(
    const Horus_code_block_output_pipe output_pipe)
    : _data_output_message_begin_available(false)
    , _data_output_message_begin{output_pipe}
{
}

// =================================================================================================
// -- Other member functions -----------------------------------------------------------------------
// =================================================================================================

bool Horus_code_block_output_message_base::data_output_message_begin_available() const
{
    return _data_output_message_begin_available;
}

bool Horus_code_block_output_message_base::data_available() const
{
    return data_output_message_begin_available();
}

void Horus_code_block_output_message_base::set_data_output_message_begin()
{
    assert(!_data_output_message_begin_available);
    _data_output_message_begin_available = true;
}

void Horus_code_block_output_message_base::ensure_data_output_message_begin()
{
    if (!_data_output_message_begin_available)
    {
        _data_output_message_begin_available = true;
    }
}

const struct Horus_code_block_data_output_message_begin &
Horus_code_block_output_message_base::get_data_output_message_begin()
{
    assert(_data_output_message_begin_available);
    _data_output_message_begin_available = false;
    return _data_output_message_begin;
}
