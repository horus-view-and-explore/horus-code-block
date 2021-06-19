// This file contains Debug Sink, a C example implementation of the Horus Code
// Block C API.
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

#include "../../Horus_code_block.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =================================================================================================
// -- Internal data --------------------------------------------------------------------------------
// =================================================================================================

#define PRINT_CONTEXT_ON_WRITE_AND_READ_KEY "print-context"

#define PRINT_DATA_CONTENTS_ON_WRITE_KEY "print-data-contents"

#define TRUE_VALUE "true"

#define FALSE_VALUE "false"

// =================================================================================================
// -- Internal types -------------------------------------------------------------------------------
// =================================================================================================

#define LOG_TEXT_CAPACITY 512u

struct Log
{
    Horus_code_block_log_severity severity;

    char text[LOG_TEXT_CAPACITY + 1u];
};

#define LOG_QUEUE_CAPACITY 10u

struct User_context
{
    time_t creation_time;

    bool running;

    bool print_context_on_write_and_read;

    bool print_data_contents_on_write;

    // -- Data -------------------------------------------------------------------------------------

    struct Horus_code_block_data data;

    // -- Log data ---------------------------------------------------------------------------------

    // Circular queue
    struct Log log_queue[LOG_QUEUE_CAPACITY];
    size_t log_queue_begin;
    size_t log_queue_size;

    struct Horus_code_block_data_log log;
};

// =================================================================================================
// -- Internal functions ---------------------------------------------------------------------------
// =================================================================================================

#ifndef _WIN32

static size_t min(const size_t a, const size_t b)
{
    return ((a < b) ? a : b);
}

#endif // not _WIN32

// -- Argument validation --------------------------------------------------------------------------

// Note that this example implementation of the Horus Code Block API also checks
// the validity of the arguments.  You do not need to do this in your
// implementation.  You can safely assume that the arguments are valid.

static const char *const _error_format_ = "%s: Error: %s\n";

static bool version_out_is_valid(const char *const function_name, unsigned int *const version)
{
    if (version == NULL)
    {
        printf(_error_format_, function_name, "version == NULL");
        return false;
    }
    return true;
}

static bool context_in_is_valid(
    const char *const function_name,
    const struct Horus_code_block_context *const context)
{
    assert(function_name != NULL);
    if (context == NULL)
    {
        printf(_error_format_, function_name, "context == NULL");
        return false;
    }
    if (context->component_instance_id == NULL)
    {
        printf(_error_format_, function_name, "context->component_instance_id == NULL");
        return false;
    }
    if (context->key_value_array.array == NULL)
    {
        printf(_error_format_, function_name, "context->key_value_array.array == NULL");
        return false;
    }
    return true;
}

static bool context_in_and_user_context_out_are_valid(
    const char *const function_name,
    const struct Horus_code_block_context *const context,
    Horus_code_block_user_context **const user_context)
{
    assert(function_name != NULL);
    if (!context_in_is_valid(function_name, context))
    {
        return false;
    }
    if (user_context != &context->user_context)
    {
        printf(_error_format_, function_name, "user_context != &context->user_context");
        return false;
    }
    return true;
}

static bool context_in_and_user_context_in_are_valid(
    const char *const function_name,
    const struct Horus_code_block_context *const context)
{
    assert(function_name != NULL);
    if (!context_in_is_valid(function_name, context))
    {
        return false;
    }
    if (context->user_context == NULL)
    {
        printf(_error_format_, function_name, "context->user_context == NULL");
        return false;
    }
    return true;
}

static bool running_is_valid(
    const char *const function_name,
    const struct Horus_code_block_context *const context,
    const bool expect_running)
{
    assert((function_name != NULL) && (context != NULL) && (context->user_context != NULL));
    const struct User_context *const user_context = (struct User_context *)context->user_context;
    if (user_context->running != expect_running)
    {
        if (user_context->running)
        {
            printf(_error_format_, function_name, "user_context->running == true");
        }
        else
        {
            printf(_error_format_, function_name, "user_context->running == false");
        }
        return false;
    }
    return true;
}

static bool context_in_and_user_context_in_and_data_in_are_valid(
    const char *const function_name,
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    assert(function_name != NULL);
    if (!context_in_and_user_context_in_are_valid(function_name, context))
    {
        return false;
    }
    if (data == NULL)
    {
        printf(_error_format_, function_name, "data == NULL");
        return false;
    }
    switch (data->type)
    {
        case Horus_code_block_data_type_start:
        case Horus_code_block_data_type_stop:
        case Horus_code_block_data_type_input_message_end:
        {
            if (data->contents != NULL)
            {
                printf(_error_format_, function_name, "data->contents != NULL");
                return false;
            }
            break;
        }
        case Horus_code_block_data_type_input_message_begin:
        case Horus_code_block_data_type_grabber_utc_timestamp:
        case Horus_code_block_data_type_ascii_in:
        case Horus_code_block_data_type_video_fourcc_in:
        case Horus_code_block_data_type_ptz_stop:
        case Horus_code_block_data_type_ptz_button:
        case Horus_code_block_data_type_ptz_orientation_rpy:
        case Horus_code_block_data_type_sensor:
        {
            if (data->contents == NULL)
            {
                printf(_error_format_, function_name, "data->contents == NULL");
                return false;
            }
            break;
        }
        default:
            printf(_error_format_, function_name, "Invalid data->type");
            return false;
    }
    return true;
}

static bool data_input_message_begin_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_input_message_begin *const data_input_message_begin)
{
    assert((function_name != NULL) && (data_input_message_begin != NULL));
    switch (data_input_message_begin->input_pipe)
    {
        case Horus_code_block_input_pipe_data_0:
        case Horus_code_block_input_pipe_data_1:
        case Horus_code_block_input_pipe_data_2:
        case Horus_code_block_input_pipe_clock:
            break;
        default:
            printf(_error_format_, function_name, "Invalid data_input_message_begin->input_pipe");
            return false;
    }
    if (data_input_message_begin->sender_id == NULL)
    {
        printf(_error_format_, function_name, "data_input_message_begin->sender_id == NULL");
        return false;
    }
    if (data_input_message_begin->source_id == NULL)
    {
        printf(_error_format_, function_name, "data_input_message_begin->source_id == NULL");
        return false;
    }
    return true;
}

static bool data_grabber_utc_timestamp_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_grabber_utc_timestamp *const data_grabber_utc_timestamp)
{
    assert((function_name != NULL) && (data_grabber_utc_timestamp != NULL));
    return true;
}

static bool data_ascii_in_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_ascii_in *const data_ascii_in)
{
    assert((function_name != NULL) && (data_ascii_in != NULL));
    if ((0u < data_ascii_in->buffer_size) && (data_ascii_in->buffer == NULL))
    {
        printf(_error_format_, function_name, "data_ascii_in->buffer == NULL");
        return false;
    }
    if ((0u < data_ascii_in->stream_index_array_size) &&
        (data_ascii_in->stream_index_array == NULL))
    {
        printf(_error_format_, function_name, "data_ascii_in->stream_index_array == NULL");
        return false;
    }
    for (size_t i = 0u; i < data_ascii_in->stream_index_array_size; ++i)
    {
        if (data_ascii_in->stream_index_array[i] == NULL)
        {
            printf(_error_format_, function_name, "data_ascii_in->stream_index_array[i] == NULL");
            printf(
                "data_ascii_in->stream_index_array[%zu/%zu] == NULL\n",
                i,
                data_ascii_in->stream_index_array_size);
            return false;
        }
    }
    return true;
}

static bool data_video_fourcc_in_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_video_fourcc_in *const data_video_fourcc_in)
{
    assert((function_name != NULL) && (data_video_fourcc_in != NULL));
    if ((0u < data_video_fourcc_in->buffer_size) && (data_video_fourcc_in->buffer == NULL))
    {
        printf(_error_format_, function_name, "data_video_fourcc_in->buffer == NULL");
        return false;
    }
    if ((0u < data_video_fourcc_in->stream_index_array_size) &&
        (data_video_fourcc_in->stream_index_array == NULL))
    {
        printf(_error_format_, function_name, "data_video_fourcc_in->stream_index_array == NULL");
        return false;
    }
    for (size_t i = 0u; i < data_video_fourcc_in->stream_index_array_size; ++i)
    {
        if (data_video_fourcc_in->stream_index_array[i] == NULL)
        {
            printf(
                _error_format_,
                function_name,
                "data_video_fourcc_in->stream_index_array[i] == NULL");
            printf(
                "data_video_fourcc_in->stream_index_array[%zu/%zu] == NULL\n",
                i,
                data_video_fourcc_in->stream_index_array_size);
            return false;
        }
    }
    return true;
}

static bool data_ptz_stop_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_stop *const data_ptz_stop)
{
    assert((function_name != NULL) && (data_ptz_stop != NULL));
    return true;
}

static bool data_ptz_button_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_button *const data_ptz_button)
{
    assert((function_name != NULL) && (data_ptz_button != NULL));
    if (data_ptz_button->button_id_pressed == NULL)
    {
        printf(_error_format_, function_name, "data_ptz_button->button_id_pressed == NULL");
        return false;
    }
    return true;
}

static bool unit_is_valid(
    const char *const function_name,
    const Horus_code_block_unit *const unit,
    const char *const error_message)
{
    assert((function_name != NULL) && (error_message != NULL));
    if (unit != NULL)
    {
        switch (*unit)
        {
            case Horus_code_block_unit_unknown_unit:
            case Horus_code_block_unit_unknown:
            case Horus_code_block_unit_degree:
            case Horus_code_block_unit_radian:
            case Horus_code_block_unit_celcius:
            case Horus_code_block_unit_normalized:
            case Horus_code_block_unit_degree_sec:
            case Horus_code_block_unit_radian_sec:
            case Horus_code_block_unit_g:
            case Horus_code_block_unit_gauss:
            case Horus_code_block_unit_percentage:
            case Horus_code_block_unit_v:
            case Horus_code_block_unit_w:
            case Horus_code_block_unit_a:
            case Horus_code_block_unit_b:
            case Horus_code_block_unit_quaternion:
            case Horus_code_block_unit_mm:
            case Horus_code_block_unit_m:
            case Horus_code_block_unit_km:
            case Horus_code_block_unit_m_sec:
            case Horus_code_block_unit_kmph:
            case Horus_code_block_unit_bpm:
            case Horus_code_block_unit_px:
            case Horus_code_block_unit_usec:
            case Horus_code_block_unit_kg:
            case Horus_code_block_unit_mbar:
            case Horus_code_block_unit_rpm:
            case Horus_code_block_unit_mhz:
            case Horus_code_block_unit_gb:
            case Horus_code_block_unit_ah:
            case Horus_code_block_unit_pa:
            case Horus_code_block_unit_kpa:
            case Horus_code_block_unit_grams_sec:
            case Horus_code_block_unit_fuel_air_ratio:
            case Horus_code_block_unit_l_hour:
            case Horus_code_block_unit_sec:
            case Horus_code_block_unit_minute:
                break;
            default:
                printf(_error_format_, function_name, "*data_sensor->unit invalid");
                return false;
        }
    }
    return true;
}

static bool data_ptz_orientation_rpy_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_orientation_rpy *const data_ptz_orientation_rpy)
{
    assert((function_name != NULL) && (data_ptz_orientation_rpy != NULL));
    if (!unit_is_valid(
            function_name,
            data_ptz_orientation_rpy->roll_unit,
            "*data_ptz_orientation_rpy->roll_unit invalid") ||
        !unit_is_valid(
            function_name,
            data_ptz_orientation_rpy->pitch_unit,
            "*data_ptz_orientation_rpy->pitch_unit invalid") ||
        !unit_is_valid(
            function_name,
            data_ptz_orientation_rpy->yaw_unit,
            "*data_ptz_orientation_rpy->yaw_unit invalid"))
    {
        return false;
    }
    return true;
}

static bool data_sensor_is_valid(
    const char *const function_name,
    const struct Horus_code_block_data_sensor *const data_sensor)
{
    assert((function_name != NULL) && (data_sensor != NULL));
    if (!unit_is_valid(function_name, data_sensor->unit, "*data_sensor->unit invalid"))
    {
        return false;
    }
    switch (data_sensor->data)
    {
        case Horus_code_block_sensor_data_string:
        case Horus_code_block_sensor_data_integer:
        case Horus_code_block_sensor_data_double:
        case Horus_code_block_sensor_data_float:
        case Horus_code_block_sensor_data_long:
        case Horus_code_block_sensor_data_unsigned_integer:
        case Horus_code_block_sensor_data_unsigned_long:
            break;
        default:
            printf(_error_format_, function_name, "data_sensor->data invalid");
            return false;
    }
    if ((data_sensor->dimension_array_size > 0u) && (data_sensor->dimension_array == NULL))
    {
        printf(_error_format_, function_name, "data_sensor->dimension_array == NULL");
        return false;
    }
    size_t value_array_size = 1u;
    for (size_t i = 0u; i < data_sensor->dimension_array_size; ++i)
    {
        if (data_sensor->dimension_array[i] < 0)
        {
            printf(_error_format_, function_name, "data_sensor->dimension_array[i] < 0");
            return false;
        }
        value_array_size *= (size_t)data_sensor->dimension_array[i];
    }
    if (data_sensor->data == Horus_code_block_sensor_data_string)
    {
        if ((value_array_size > 0u) && (data_sensor->string_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->string_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->string_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->string_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_integer)
    {
        if ((value_array_size > 0u) && (data_sensor->integer_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->integer_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->integer_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->integer_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_double)
    {
        if ((value_array_size > 0u) && (data_sensor->double_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->double_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->double_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->double_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_float)
    {
        if ((value_array_size > 0u) && (data_sensor->float_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->float_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->float_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->float_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_long)
    {
        if ((value_array_size > 0u) && (data_sensor->long_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->long_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->long_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->long_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_unsigned_integer)
    {
        if ((value_array_size > 0u) && (data_sensor->unsigned_integer_value_array == NULL))
        {
            printf(
                _error_format_, function_name, "data_sensor->unsigned_integer_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->unsigned_integer_value_array != NULL)
        {
            printf(
                _error_format_, function_name, "data_sensor->unsigned_integer_value_array != NULL");
            return false;
        }
    }
    if (data_sensor->data == Horus_code_block_sensor_data_unsigned_long)
    {
        if ((value_array_size > 0u) && (data_sensor->unsigned_long_value_array == NULL))
        {
            printf(_error_format_, function_name, "data_sensor->unsigned_long_value_array == NULL");
            return false;
        }
    }
    else
    {
        if (data_sensor->unsigned_long_value_array != NULL)
        {
            printf(_error_format_, function_name, "data_sensor->unsigned_long_value_array != NULL");
            return false;
        }
    }
    if ((data_sensor->metadata_dimension_array_size > 0u) &&
        (data_sensor->metadata_dimension_array == NULL))
    {
        printf(_error_format_, function_name, "data_sensor->metadata_dimension_array == NULL");
        return false;
    }
    size_t metadata_array_size = 1u;
    for (size_t i = 0u; i < data_sensor->metadata_dimension_array_size; ++i)
    {
        if (data_sensor->metadata_dimension_array[i] < 0)
        {
            printf(_error_format_, function_name, "data_sensor->metadata_dimension_array[i] < 0");
            return false;
        }
        metadata_array_size *= (size_t)data_sensor->metadata_dimension_array[i];
    }
    if ((metadata_array_size > 0u) && (data_sensor->metadata_array == NULL))
    {
        printf(_error_format_, function_name, "data_sensor->metadata_array == NULL");
        return false;
    }
    return true;
}

// -- Printing -------------------------------------------------------------------------------------

static void print_start(const char *const function_name)
{
    assert(function_name != NULL);
    printf("%s: Start.\n", function_name);
}

static void print_success(const char *const function_name)
{
    assert(function_name != NULL);
    printf("%s: Success.\n", function_name);
}

static void print_key_value(
    const char *const key_value_prefix,
    const struct Horus_code_block_key_value key_value)
{
    assert((key_value_prefix != NULL) && (key_value.key != NULL) && (key_value.value != NULL));
    printf("%s{ key = \"%s\" value = \"%s\" }\n", key_value_prefix, key_value.key, key_value.value);
}

static void print_key_value_array(
    const char *const key_value_array_prefix,
    const struct Horus_code_block_key_value_array *const key_value_array)
{
    assert(
        (key_value_array_prefix != NULL) && (key_value_array != NULL) &&
        (key_value_array->array != NULL));
    puts("{");
    const char *const extra_prefix = "  ";
    char *const key_value_prefix =
        malloc(strlen(key_value_array_prefix) + strlen(extra_prefix) + 1);
    strcpy(key_value_prefix, key_value_array_prefix);
    strcat(key_value_prefix, extra_prefix);
    for (size_t i = 0u; i < key_value_array->size; ++i)
    {
        print_key_value(key_value_prefix, key_value_array->array[i]);
    }
    free(key_value_prefix);
    printf("%s}\n", key_value_array_prefix);
}

static void print_user_context(
    const char *const user_context_prefix,
    const struct User_context *const user_context)
{
    assert((user_context_prefix != NULL) && (user_context != NULL));
    printf(
        "{\n"
        "%s  creation_time = %jd\n"
        "%s  running = %s\n"
        "%s  print_context_on_write_and_read = %s\n"
        "%s  print_data_contents_on_write = %s\n"
        "%s  log_queue_begin = %ju\n"
        "%s  log_queue_size = %ju\n"
        "%s}\n",
        user_context_prefix,
        (intmax_t)user_context->creation_time,
        user_context_prefix,
        user_context->running ? TRUE_VALUE : FALSE_VALUE,
        user_context_prefix,
        user_context->print_context_on_write_and_read ? TRUE_VALUE : FALSE_VALUE,
        user_context_prefix,
        user_context->print_data_contents_on_write ? TRUE_VALUE : FALSE_VALUE,
        user_context_prefix,
        (uintmax_t)user_context->log_queue_begin,
        user_context_prefix,
        (uintmax_t)user_context->log_queue_size,
        user_context_prefix);
}

static void
print_context(const char *const function_name, const struct Horus_code_block_context *const context)
{
    assert(
        (function_name != NULL) && (context != NULL) && (context->component_instance_id != NULL));
    printf("%s: context = {\n", function_name);
    const char *const prefix = "  ";
    if (context->user_context != NULL)
    {
        printf("%suser_context = ", prefix);
        print_user_context(prefix, (struct User_context *)context->user_context);
    }
    printf(
        "%scomponent_instance_id = \"%s\"\n"
        "%skey_value_array = ",
        prefix,
        context->component_instance_id,
        prefix);
    print_key_value_array(prefix, &context->key_value_array);
    puts("}");
}

static void
print_data_type(const char *const function_name, const Horus_code_block_data_type data_type)
{
    assert(function_name != NULL);
    static const char *const format = "%s: data_type_%s.\n";
    switch (data_type)
    {
        case Horus_code_block_data_type_start:
            printf(format, function_name, "start");
            break;
        case Horus_code_block_data_type_stop:
            printf(format, function_name, "stop");
            break;
        case Horus_code_block_data_type_input_message_begin:
            printf(format, function_name, "input_message_begin");
            break;
        case Horus_code_block_data_type_input_message_end:
            printf(format, function_name, "input_message_end");
            break;
        case Horus_code_block_data_type_grabber_utc_timestamp:
            printf(format, function_name, "grabber_utc_timestamp");
            break;
        case Horus_code_block_data_type_ascii_in:
            printf(format, function_name, "ascii_in");
            break;
        case Horus_code_block_data_type_video_fourcc_in:
            printf(format, function_name, "video_fourcc_in");
            break;
        case Horus_code_block_data_type_ptz_stop:
            printf(format, function_name, "ptz_stop");
            break;
        case Horus_code_block_data_type_ptz_button:
            printf(format, function_name, "ptz_button");
            break;
        case Horus_code_block_data_type_ptz_orientation_rpy:
            printf(format, function_name, "ptz_orientation_rpy");
            break;
        case Horus_code_block_data_type_sensor:
            printf(format, function_name, "sensor");
            break;
        default:
            assert(false);
    }
}

static void print_data_input_message_begin(
    const char *const function_name,
    const struct Horus_code_block_data_input_message_begin *const data_input_message_begin)
{
    assert(
        (function_name != NULL) && (data_input_message_begin != NULL) &&
        (data_input_message_begin->source_id != NULL));
    printf(
        "%s: data_input_message_begin = {\n"
        "  input_pipe = ",
        function_name);
    switch (data_input_message_begin->input_pipe)
    {
        case Horus_code_block_input_pipe_data_0:
            puts("data_0");
            break;
        case Horus_code_block_input_pipe_data_1:
            puts("data_1");
            break;
        case Horus_code_block_input_pipe_data_2:
            puts("data_2");
            break;
        case Horus_code_block_input_pipe_clock:
            puts("clock");
            break;
        default:
            assert(false);
    }
    printf("  sender_id = %s\n", data_input_message_begin->sender_id);
    printf("  source_id = %s\n", data_input_message_begin->source_id);
    puts("}");
}

static void print_data_grabber_utc_timestamp(
    const char *const function_name,
    const struct Horus_code_block_data_grabber_utc_timestamp *const data_grabber_utc_timestamp)
{
    assert((function_name != NULL) && (data_grabber_utc_timestamp != NULL));
    printf("%s: data_grabber_utc_timestamp = {\n", function_name);
    if (data_grabber_utc_timestamp->system_coordinate != NULL)
    {
        printf(
            "  system_coordinate = %s\n",
            *data_grabber_utc_timestamp->system_coordinate ? TRUE_VALUE : FALSE_VALUE);
    }
    printf("  stamp = %ju\n", (uintmax_t)data_grabber_utc_timestamp->stamp);
    puts("}");
}

static void print_data_ascii_in(
    const char *const function_name,
    const struct Horus_code_block_data_ascii_in *const data_ascii_in)
{
    assert(
        (function_name != NULL) && (data_ascii_in != NULL) &&
        ((data_ascii_in->buffer_size == 0u) || (data_ascii_in->buffer != NULL)) &&
        ((data_ascii_in->stream_index_array_size == 0u) ||
         (data_ascii_in->stream_index_array != NULL)));
    printf("%s: data_ascii_in = {\n", function_name);
    if (data_ascii_in->data_id != NULL)
    {
        printf("  data_id = %s\n", data_ascii_in->data_id);
    }
    printf("  buffer_size = %zu\n", data_ascii_in->buffer_size);
    if (0u < data_ascii_in->buffer_size)
    {
        printf("  buffer = %.*s\n", (int)data_ascii_in->buffer_size, data_ascii_in->buffer);
    }
    puts("  stream_index_array = {");
    for (size_t i = 0u; i < data_ascii_in->stream_index_array_size; ++i)
    {
        assert(data_ascii_in->stream_index_array[i] != NULL);
        printf("    %zu = %s\n", i, data_ascii_in->stream_index_array[i]);
    }
    puts("  }");
    puts("}");
}

static void print_data_video_fourcc_in(
    const char *const function_name,
    const struct Horus_code_block_data_video_fourcc_in *const data_video_fourcc_in)
{
    assert(
        (function_name != NULL) && (data_video_fourcc_in != NULL) &&
        ((data_video_fourcc_in->buffer_size == 0u) || (data_video_fourcc_in->buffer != NULL)) &&
        ((data_video_fourcc_in->stream_index_array_size == 0u) ||
         (data_video_fourcc_in->stream_index_array != NULL)));
    printf("%s: data_video_fourcc_in = {\n", function_name);
    if (data_video_fourcc_in->data_id != NULL)
    {
        printf("  data_id = %s\n", data_video_fourcc_in->data_id);
    }
    printf("  buffer_size = %zu\n", data_video_fourcc_in->buffer_size);
    puts("  stream_index_array = {");
    for (size_t i = 0u; i < data_video_fourcc_in->stream_index_array_size; ++i)
    {
        assert(data_video_fourcc_in->stream_index_array[i] != NULL);
        printf("    %zu = %s\n", i, data_video_fourcc_in->stream_index_array[i]);
    }
    puts("  }");
    printf("  fourcc = %.*s\n", 4, data_video_fourcc_in->fourcc);
    if (data_video_fourcc_in->width != NULL)
    {
        printf("  width = %ju\n", (uintmax_t)*data_video_fourcc_in->width);
    }
    if (data_video_fourcc_in->height != NULL)
    {
        printf("  height = %ju\n", (uintmax_t)*data_video_fourcc_in->height);
    }
    if (data_video_fourcc_in->is_key_frame != NULL)
    {
        printf(
            "  is_key_frame = %s\n",
            *data_video_fourcc_in->is_key_frame ? TRUE_VALUE : FALSE_VALUE);
    }
    if (data_video_fourcc_in->line_stride != NULL)
    {
        printf("  line_stride = %ju\n", (uintmax_t)*data_video_fourcc_in->line_stride);
    }
    puts("}");
}

static void print_data_ptz_stop(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_stop *const data_ptz_stop)
{
    assert((function_name != NULL) && (data_ptz_stop != NULL));
    printf("%s: data_ptz_stop = {\n", function_name);
    if (data_ptz_stop->id != NULL)
    {
        printf("  id = %s\n", data_ptz_stop->id);
    }
    printf("  stop = %s\n", data_ptz_stop->stop ? TRUE_VALUE : FALSE_VALUE);
    puts("}");
}

static void print_data_ptz_button(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_button *const data_ptz_button)
{
    assert(
        (function_name != NULL) && (data_ptz_button != NULL) &&
        (data_ptz_button->button_id_pressed != NULL));
    printf("%s: data_ptz_button = {\n", function_name);
    if (data_ptz_button->id != NULL)
    {
        printf("  id = %s\n", data_ptz_button->id);
    }
    printf("  button_id_pressed = %s\n", data_ptz_button->button_id_pressed);
    puts("}");
}

static void print_unit(const Horus_code_block_unit *const unit, const char *const unit_name)
{
    assert(unit_name != NULL);
    if (unit != NULL)
    {
        switch (*unit)
        {
            case Horus_code_block_unit_unknown_unit:
                printf("  %s = unknown_unit\n", unit_name);
                break;
            case Horus_code_block_unit_unknown:
                printf("  %s = unknown\n", unit_name);
                break;
            case Horus_code_block_unit_degree:
                printf("  %s = degree\n", unit_name);
                break;
            case Horus_code_block_unit_radian:
                printf("  %s = radian\n", unit_name);
                break;
            case Horus_code_block_unit_celcius:
                printf("  %s = celcius\n", unit_name);
                break;
            case Horus_code_block_unit_normalized:
                printf("  %s = normalized\n", unit_name);
                break;
            case Horus_code_block_unit_degree_sec:
                printf("  %s = degree_sec\n", unit_name);
                break;
            case Horus_code_block_unit_radian_sec:
                printf("  %s = radian_sec\n", unit_name);
                break;
            case Horus_code_block_unit_g:
                printf("  %s = g\n", unit_name);
                break;
            case Horus_code_block_unit_gauss:
                printf("  %s = gauss\n", unit_name);
                break;
            case Horus_code_block_unit_percentage:
                printf("  %s = percentage\n", unit_name);
                break;
            case Horus_code_block_unit_v:
                printf("  %s = v\n", unit_name);
                break;
            case Horus_code_block_unit_w:
                printf("  %s = w\n", unit_name);
                break;
            case Horus_code_block_unit_a:
                printf("  %s = a\n", unit_name);
                break;
            case Horus_code_block_unit_b:
                printf("  %s = b\n", unit_name);
                break;
            case Horus_code_block_unit_quaternion:
                printf("  %s = quaternion\n", unit_name);
                break;
            case Horus_code_block_unit_mm:
                printf("  %s = mm\n", unit_name);
                break;
            case Horus_code_block_unit_m:
                printf("  %s = m\n", unit_name);
                break;
            case Horus_code_block_unit_km:
                printf("  %s = km\n", unit_name);
                break;
            case Horus_code_block_unit_m_sec:
                printf("  %s = m_sec\n", unit_name);
                break;
            case Horus_code_block_unit_kmph:
                printf("  %s = kmph\n", unit_name);
                break;
            case Horus_code_block_unit_bpm:
                printf("  %s = bpm\n", unit_name);
                break;
            case Horus_code_block_unit_px:
                printf("  %s = px\n", unit_name);
                break;
            case Horus_code_block_unit_usec:
                printf("  %s = usec\n", unit_name);
                break;
            case Horus_code_block_unit_kg:
                printf("  %s = kg\n", unit_name);
                break;
            case Horus_code_block_unit_mbar:
                printf("  %s = mbar\n", unit_name);
                break;
            case Horus_code_block_unit_rpm:
                printf("  %s = rpm\n", unit_name);
                break;
            case Horus_code_block_unit_mhz:
                printf("  %s = mhz\n", unit_name);
                break;
            case Horus_code_block_unit_gb:
                printf("  %s = gb\n", unit_name);
                break;
            case Horus_code_block_unit_ah:
                printf("  %s = ah\n", unit_name);
                break;
            case Horus_code_block_unit_pa:
                printf("  %s = pa\n", unit_name);
                break;
            case Horus_code_block_unit_kpa:
                printf("  %s = kpa\n", unit_name);
                break;
            case Horus_code_block_unit_grams_sec:
                printf("  %s = grams_sec\n", unit_name);
                break;
            case Horus_code_block_unit_fuel_air_ratio:
                printf("  %s = fuel_air_ratio\n", unit_name);
                break;
            case Horus_code_block_unit_l_hour:
                printf("  %s = l_hour\n", unit_name);
                break;
            case Horus_code_block_unit_sec:
                printf("  %s = sec\n", unit_name);
                break;
            case Horus_code_block_unit_minute:
                printf("  %s = minute\n", unit_name);
                break;
            default:
                assert(false);
        }
    }
}

static void print_data_ptz_orientation_rpy(
    const char *const function_name,
    const struct Horus_code_block_data_ptz_orientation_rpy *const data_ptz_orientation_rpy)
{
    assert((function_name != NULL) && (data_ptz_orientation_rpy != NULL));
    printf("%s: data_ptz_orientation_rpy = {\n", function_name);
    if (data_ptz_orientation_rpy->id != NULL)
    {
        printf("  id = %s\n", data_ptz_orientation_rpy->id);
    }
    printf("  roll = %f\n", data_ptz_orientation_rpy->roll);
    print_unit(data_ptz_orientation_rpy->roll_unit, "roll_unit");
    printf("  pitch = %f\n", data_ptz_orientation_rpy->pitch);
    print_unit(data_ptz_orientation_rpy->pitch_unit, "pitch_unit");
    printf("  yaw = %f\n", data_ptz_orientation_rpy->yaw);
    print_unit(data_ptz_orientation_rpy->yaw_unit, "yaw_unit");
    puts("}");
}

static void print_data_sensor(
    const char *const function_name,
    const struct Horus_code_block_data_sensor *const data_sensor)
{
    assert((function_name != NULL) && (data_sensor != NULL));
    printf("%s: data_sensor = {\n", function_name);
    if (data_sensor->name != NULL)
    {
        printf("  name = %s\n", data_sensor->name);
    }
    print_unit(data_sensor->unit, "unit");
    switch (data_sensor->data)
    {
        case Horus_code_block_sensor_data_string:
            puts("  data = string");
            break;
        case Horus_code_block_sensor_data_integer:
            puts("  data = integer");
            break;
        case Horus_code_block_sensor_data_double:
            puts("  data = double");
            break;
        case Horus_code_block_sensor_data_float:
            puts("  data = float");
            break;
        case Horus_code_block_sensor_data_long:
            puts("  data = long");
            break;
        case Horus_code_block_sensor_data_unsigned_integer:
            puts("  data = unsigned_integer");
            break;
        case Horus_code_block_sensor_data_unsigned_long:
            puts("  data = unsigned_long");
            break;
        default:
            assert(false);
    }
    printf("  dimension_array_size = %zu\n", data_sensor->dimension_array_size);
    puts("  dimension_array = {");
    size_t value_array_size = 1u;
    for (size_t i = 0u; i < data_sensor->dimension_array_size; ++i)
    {
        printf("    %jd\n", (intmax_t)data_sensor->dimension_array[i]);
        value_array_size *= (size_t)data_sensor->dimension_array[i];
    }
    puts("  }");
    switch (data_sensor->data)
    {
        case Horus_code_block_sensor_data_string:
            puts("  string_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %s\n", data_sensor->string_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_integer:
            puts("  integer_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %jd\n", (intmax_t)data_sensor->integer_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_double:
            puts("  double_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %g\n", data_sensor->double_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_float:
            puts("  float_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %g\n", data_sensor->float_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_long:
            puts("  long_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %jd\n", (intmax_t)data_sensor->long_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_unsigned_integer:
            puts("  unsigned_integer_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %ju\n", (uintmax_t)data_sensor->unsigned_integer_value_array[i]);
            }
            puts("  }");
            break;
        case Horus_code_block_sensor_data_unsigned_long:
            puts("  unsigned_long_value_array = {");
            for (size_t i = 0u; i < value_array_size; ++i)
            {
                printf("    %ju\n", (uintmax_t)data_sensor->unsigned_long_value_array[i]);
            }
            puts("  }");
            break;
        default:
            assert(false);
    }
    printf("  metadata_dimension_array_size = %zu\n", data_sensor->metadata_dimension_array_size);
    puts("  metadata_dimension_array = {");
    size_t metadata_array_size = 1u;
    for (size_t i = 0u; i < data_sensor->metadata_dimension_array_size; ++i)
    {
        printf("    %jd\n", (intmax_t)data_sensor->metadata_dimension_array[i]);
        metadata_array_size *= (size_t)data_sensor->metadata_dimension_array[i];
    }
    puts("  }");
    puts("  metadata_array = {");
    for (size_t i = 0u; i < metadata_array_size; ++i)
    {
        printf("    %s\n", data_sensor->metadata_array[i]);
    }
    puts("  }");
    puts("}");
}

// -- Log queue ------------------------------------------------------------------------------------

static bool log_queue_empty(struct User_context *const user_context)
{
    assert(
        (user_context != NULL) && (0u <= user_context->log_queue_begin) &&
        (user_context->log_queue_begin < LOG_QUEUE_CAPACITY) &&
        (user_context->log_queue_size <= LOG_QUEUE_CAPACITY));
    return (user_context->log_queue_size == 0u);
}

static void log_queue_push(
    const char *const function_name,
    struct User_context *const user_context,
    const Horus_code_block_log_severity severity,
    const char *const text)
{
    assert(
        (function_name != NULL) && (user_context != NULL) &&
        (0u <= user_context->log_queue_begin) &&
        (user_context->log_queue_begin < LOG_QUEUE_CAPACITY) &&
        (user_context->log_queue_size <= LOG_QUEUE_CAPACITY));
    if (user_context->log_queue_size == LOG_QUEUE_CAPACITY)
    {
        printf(_error_format_, function_name, "user_context->log_queue_size == LOG_QUEUE_CAPACITY");
    }
    else
    {
        struct Log *const log =
            user_context->log_queue +
            ((user_context->log_queue_begin + user_context->log_queue_size) % LOG_QUEUE_CAPACITY);
        log->severity = severity;
        strncpy(log->text, text, min(LOG_TEXT_CAPACITY, strlen(text) + 1u));
        log->text[LOG_TEXT_CAPACITY] = '\0';
        ++user_context->log_queue_size;
    }
}

static const struct Log *
log_queue_pop(const char *const function_name, struct User_context *const user_context)
{
    assert(
        (function_name != NULL) && (user_context != NULL) &&
        (0u <= user_context->log_queue_begin) &&
        (user_context->log_queue_begin < LOG_QUEUE_CAPACITY) &&
        (0u < user_context->log_queue_size) &&
        (user_context->log_queue_size <= LOG_QUEUE_CAPACITY));
    if (user_context->log_queue_size == 0u)
    {
        printf(_error_format_, function_name, "user_context->log_queue_size == 0u");
        return NULL;
    }
    else
    {
        struct Log *const log = user_context->log_queue + user_context->log_queue_begin;
        ++user_context->log_queue_begin;
        user_context->log_queue_begin %= LOG_QUEUE_CAPACITY;
        --user_context->log_queue_size;
        return log;
    }
}

// -- Key-value pairs ------------------------------------------------------------------------------

static const char *find_value(
    const struct Horus_code_block_key_value_array *const key_value_array,
    const char *const key)
{
    assert((key_value_array != NULL) && (key_value_array->array != NULL));
    for (size_t i = 0u; i < key_value_array->size; ++i)
    {
        const struct Horus_code_block_key_value key_value = key_value_array->array[i];
        assert((key_value.key != NULL) && (key_value.value != NULL));
        if (strcmp(key_value.key, key) == 0)
        {
            return key_value.value;
        }
    }
    return NULL;
}

static bool get_optional_bool_value(
    const char *const function_name,
    const struct Horus_code_block_context *const context,
    const char *const bool_key,
    const bool default_bool_value)
{
    const char *const bool_value = find_value(&context->key_value_array, bool_key);
    if (bool_value != NULL)
    {
        if (strcmp(bool_value, TRUE_VALUE) == 0)
        {
            return true;
        }
        else if (strcmp(bool_value, FALSE_VALUE) == 0)
        {
            return false;
        }
        else
        {
            char text[LOG_TEXT_CAPACITY + 1u];
            snprintf(
                text,
                sizeof(text),
                "Key '%s' has invalid value '%s', using default value '%s' instead.",
                bool_key,
                bool_value,
                default_bool_value ? TRUE_VALUE : FALSE_VALUE);
            log_queue_push(
                function_name,
                (struct User_context *)context->user_context,
                Horus_code_block_log_severity_warning,
                text);
            return default_bool_value;
        }
    }
    else
    {
        char text[LOG_TEXT_CAPACITY + 1u];
        snprintf(
            text,
            sizeof(text),
            "Key '%s' not found, using default value '%s' instead.",
            bool_key,
            default_bool_value ? TRUE_VALUE : FALSE_VALUE);
        log_queue_push(
            function_name,
            (struct User_context *)context->user_context,
            Horus_code_block_log_severity_info,
            text);
        return default_bool_value;
    }
}

// =================================================================================================
// -- API ------------------------------------------------------------------------------------------
// =================================================================================================

// -- Mandatory version function -------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    const char *const function_name = "horus_code_block_get_version";
    print_start(function_name);

    if (!version_out_is_valid(function_name, version))
    {
        return Horus_code_block_error;
    }

    *version = HORUS_CODE_BLOCK_VERSION;

    print_success(function_name);
    return Horus_code_block_success;
}

// -- Optional discovery function ------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    const char *const function_name = "horus_code_block_get_discovery_info";
    print_start(function_name);

    static const struct Horus_code_block_discovery_info static_discovery_info = {
        .name = "Debug Sink",
        .description =
            "Print information on the shared library calls.\n"
            "\n"
            "You can control the amount of information that Debug Sink prints by specifying "
            "certain key-value pairs. The values can be either '" TRUE_VALUE "' (the default) "
            "or '" FALSE_VALUE "'.\n"
            "\n"
            "'" PRINT_CONTEXT_ON_WRITE_AND_READ_KEY "': When '" TRUE_VALUE "' print the context "
            "upon entry and exit of calls to horus_code_block_write() and "
            "horus_code_block_read().\n"
            "'" PRINT_DATA_CONTENTS_ON_WRITE_KEY "': When '" TRUE_VALUE "' print the contents "
            "of data written using horus_code_block_write(). When '" FALSE_VALUE "' print just "
            "the type of data."};

    *discovery_info = &static_discovery_info;

    print_success(function_name);
    return Horus_code_block_success;
}

// -- Mandatory functions --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const context,
    Horus_code_block_user_context **const user_context)
{
    const char *const function_name = "horus_code_block_open";
    print_start(function_name);

    if (!context_in_and_user_context_out_are_valid(function_name, context, user_context))
    {
        return Horus_code_block_error;
    }

    print_context(function_name, context);

    struct User_context *const user_context_ = malloc(sizeof(struct User_context));

    user_context_->creation_time = time(NULL);
    user_context_->running = false;
    user_context_->log_queue_begin = 0u;
    user_context_->log_queue_size = 0u;
    *user_context = user_context_;

    log_queue_push(function_name, user_context_, Horus_code_block_log_severity_info, "Opened.");

    // This may push to the log queue.
    user_context_->print_context_on_write_and_read =
        get_optional_bool_value(function_name, context, PRINT_CONTEXT_ON_WRITE_AND_READ_KEY, true);
    user_context_->print_data_contents_on_write =
        get_optional_bool_value(function_name, context, PRINT_DATA_CONTENTS_ON_WRITE_KEY, true);

    print_context(function_name, context);

    print_success(function_name);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    const char *const function_name = "horus_code_block_close";
    print_start(function_name);

    if (!context_in_and_user_context_in_are_valid(function_name, context))
    {
        return Horus_code_block_error;
    }
    const bool expect_running = false;
    if (!running_is_valid(function_name, context, expect_running))
    {
        return Horus_code_block_error;
    }

    print_context(function_name, context);

    free(context->user_context);

    print_success(function_name);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    const char *const function_name = "horus_code_block_write";
    print_start(function_name);

    if (!context_in_and_user_context_in_and_data_in_are_valid(function_name, context, data))
    {
        return Horus_code_block_error;
    }

    struct User_context *const user_context = (struct User_context *)context->user_context;

    if (user_context->print_context_on_write_and_read)
    {
        print_context(function_name, context);
    }

    assert(data != NULL);
    // Check running.
    switch (data->type)
    {
        case Horus_code_block_data_type_start:
        {
            const bool expect_running = false;
            if (!running_is_valid(function_name, context, expect_running))
            {
                return Horus_code_block_error;
            }
            break;
        }
        case Horus_code_block_data_type_stop:
        case Horus_code_block_data_type_input_message_begin:
        case Horus_code_block_data_type_input_message_end:
        case Horus_code_block_data_type_grabber_utc_timestamp:
        case Horus_code_block_data_type_ascii_in:
        case Horus_code_block_data_type_video_fourcc_in:
        case Horus_code_block_data_type_ptz_stop:
        case Horus_code_block_data_type_ptz_button:
        case Horus_code_block_data_type_ptz_orientation_rpy:
        case Horus_code_block_data_type_sensor:
        {
            const bool expect_running = true;
            if (!running_is_valid(function_name, context, expect_running))
            {
                return Horus_code_block_error;
            }
            break;
        }
        default:
            assert(false);
    }
    // Check data validity.
    switch (data->type)
    {
        case Horus_code_block_data_type_start:
        case Horus_code_block_data_type_stop:
        case Horus_code_block_data_type_input_message_end:
            assert(data->contents == NULL);
            break;
        case Horus_code_block_data_type_input_message_begin:
            assert(data->contents != NULL);
            if (!data_input_message_begin_is_valid(
                    function_name,
                    (const struct Horus_code_block_data_input_message_begin *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_grabber_utc_timestamp:
            assert(data->contents != NULL);
            if (!data_grabber_utc_timestamp_is_valid(
                    function_name,
                    (const struct Horus_code_block_data_grabber_utc_timestamp *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_ascii_in:
            assert(data->contents != NULL);
            if (!data_ascii_in_is_valid(
                    function_name, (const struct Horus_code_block_data_ascii_in *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_video_fourcc_in:
            assert(data->contents != NULL);
            if (!data_video_fourcc_in_is_valid(
                    function_name,
                    (const struct Horus_code_block_data_video_fourcc_in *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_ptz_stop:
            assert(data->contents != NULL);
            if (!data_ptz_stop_is_valid(
                    function_name, (const struct Horus_code_block_data_ptz_stop *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_ptz_button:
            assert(data->contents != NULL);
            if (!data_ptz_button_is_valid(
                    function_name, (const struct Horus_code_block_data_ptz_button *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_ptz_orientation_rpy:
            assert(data->contents != NULL);
            if (!data_ptz_orientation_rpy_is_valid(
                    function_name,
                    (const struct Horus_code_block_data_ptz_orientation_rpy *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        case Horus_code_block_data_type_sensor:
            assert(data->contents != NULL);
            if (!data_sensor_is_valid(
                    function_name, (const struct Horus_code_block_data_sensor *)data->contents))
            {
                return Horus_code_block_error;
            }
            break;
        default:
            assert(false);
    }
    // Print data (and start/stop/log).
    switch (data->type)
    {
        case Horus_code_block_data_type_start:
            assert(data->contents == NULL);
            print_data_type(function_name, data->type);
            user_context->running = true;
            log_queue_push(
                function_name, user_context, Horus_code_block_log_severity_info, "Started.");
            break;
        case Horus_code_block_data_type_stop:
            assert(data->contents == NULL);
            print_data_type(function_name, data->type);
            user_context->running = false;
            log_queue_push(
                function_name, user_context, Horus_code_block_log_severity_info, "Stopped.");
            break;
        case Horus_code_block_data_type_input_message_begin:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_input_message_begin(
                    function_name,
                    (const struct Horus_code_block_data_input_message_begin *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_input_message_end:
            assert(data->contents == NULL);
            print_data_type(function_name, data->type);
            break;
        case Horus_code_block_data_type_grabber_utc_timestamp:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_grabber_utc_timestamp(
                    function_name,
                    (const struct Horus_code_block_data_grabber_utc_timestamp *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_ascii_in:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_ascii_in(
                    function_name, (const struct Horus_code_block_data_ascii_in *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_video_fourcc_in:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_video_fourcc_in(
                    function_name,
                    (const struct Horus_code_block_data_video_fourcc_in *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_ptz_stop:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_ptz_stop(
                    function_name, (const struct Horus_code_block_data_ptz_stop *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_ptz_button:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_ptz_button(
                    function_name, (const struct Horus_code_block_data_ptz_button *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_ptz_orientation_rpy:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_ptz_orientation_rpy(
                    function_name,
                    (const struct Horus_code_block_data_ptz_orientation_rpy *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        case Horus_code_block_data_type_sensor:
            assert(data->contents != NULL);
            if (user_context->print_data_contents_on_write)
            {
                print_data_sensor(
                    function_name, (const struct Horus_code_block_data_sensor *)data->contents);
            }
            else
            {
                print_data_type(function_name, data->type);
            }
            break;
        default:
            assert(false);
    }

    if (user_context->print_context_on_write_and_read)
    {
        print_context(function_name, context);
    }

    print_success(function_name);
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    const char *const function_name = "horus_code_block_read";
    print_start(function_name);

    if (!context_in_and_user_context_in_are_valid(function_name, context))
    {
        return Horus_code_block_error;
    }

    struct User_context *const user_context = (struct User_context *)context->user_context;

    if (user_context->print_context_on_write_and_read)
    {
        print_context(function_name, context);
    }

    if (!log_queue_empty(user_context))
    {
        user_context->data.type = Horus_code_block_data_type_log;
        const struct Log *const log = log_queue_pop(function_name, user_context);
        assert(log != NULL);
        user_context->log.severity = log->severity;
        user_context->log.text = log->text;
        user_context->data.contents = &user_context->log;
        *data = &user_context->data;
    }
    Horus_code_block_result result = Horus_code_block_success;
    if (!log_queue_empty(user_context))
    {
        result |= Horus_code_block_read_flag_read_more;
    }

    if (user_context->print_context_on_write_and_read)
    {
        print_context(function_name, context);
    }

    print_success(function_name);
    return result;
}
