// This file contains the Horus Code Block C API.
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

/// @file
///
/// The Horus Code Block C API.
///
/// By implementing the functions in this header file using either C or C++ and
/// compiling/linking it into a shared library, you can create your own Horus
/// System V2 component by loading that shared library into the Horus Code Block
/// component.
///
/// A Horus Code Block component can be used directly by using its "Shared
/// Library" property to specify the path to a shared library implementing the
/// Horus Code Block C API .  A Horus Code Block component can also be
/// discovered by System V2.  The Horus Code Block discoverer looks for shared
/// libraries implementing the C API in files or directories specified by the
/// "--code-block-shared-library-path" program option.  When implemented, the
/// discoverer calls the horus_code_block_get_discovery_info() function of the
/// shared library to populate the name and description of the discovered Horus
/// Code Block component.
///
/// Upon initialization, the Horus Code Block component first loads your shared
/// library.  It then calls horus_code_block_open().  This function can be used
/// to create an arbitrary user context.  This user context can be accessed
/// through the user_context member of the struct Horus_code_block_context that
/// is passed to all subsequent API calls.  When the component is destroyed, it
/// calls horus_code_block_close() which should destroy the user context again.
///
/// The other two API functions are horus_code_block_write() and
/// horus_code_block_read() are used to transfer data between the Horus Code
/// Block component and its shared library.  When the Horus Code Block component
/// is run, it calls horus_code_block_write() passing
/// Horus_code_block_data_type_start when starting and horus_code_block_write()
/// passing passing Horus_code_block_data_type_stop when stopping.
///
/// When the Horus Code Block component is running, an input message that
/// arrives at one of the input pipes of the component will trigger a number of
/// calls to horus_code_block_write() followed by a number of calls to
/// horus_code_block_read().  The writes are used to transfer the data in the
/// input message from the component to the shared library and the reads can be
/// used by the shared library to compose an output message that is sent by the
/// component using one of its output pipes.  The reads can also be used to send
/// log data (Horus_code_block_data_type_log) to the component for it to log.
/// If a shared library is only interested in a subset of the data found in
/// input messages, it can limit the calls to horus_code_block_write() to those
/// data types by using horus_code_block_read() to pass subscription data
/// (Horus_code_block_data_subscriptions) to the component.  By default, the
/// Data input pipes are subscribed to all data types, while the Clock input
/// pipe is subscribed only to Horus_code_block_data_grabber_utc_timestamp.
///
/// Below is a more detailed overview of the control flow during the various
/// phases that the shared library is used.
///
///
/// Control flow during discovery
///
/// DISCOVERER                                       SHARED LIBRARY
///
/// Find shared library file
///   using --code-block-shared-library-path
/// Open shared library file
/// Call horus_code_block_get_version()        --->
///                                            <---  Return HORUS_CODE_BLOCK_VERSION
/// Call horus_code_block_get_discovery_info() --->
///                                            <---  Return discovery info
/// Close shared library file
///
///
/// Control flow during Horus Code Block component initialization
///
/// COMPONENT                                        SHARED LIBRARY
///
/// Open shared library file
/// Call horus_code_block_get_version()        --->
///                                            <---  Return HORUS_CODE_BLOCK_VERSION
/// Call horus_code_block_open()               --->  Create user context
///                                            <---  Return user context
/// Call horus_code_block_read() until done    --->
///                                            <---  Return subscriptions
///                                            <---  Return logs
///
///
/// Control flow during Horus Code Block component start
///
/// COMPONENT                                        SHARED LIBRARY
///
/// Call horus_code_block_write()              --->  Receive start
/// Call horus_code_block_read() until done    --->
///                                            <---  Return subscriptions
///                                            <---  Return logs
///
///
/// Control flow during Horus Code Block component input message reception
///
/// COMPONENT                                        SHARED LIBRARY
///
/// Lock input message for reading
/// Call horus_code_block_write()              --->  Receive input message begin
/// Call horus_code_block_write() for each
///   subscribed input message data type       --->  Receive input message data
/// Call horus_code_block_write()              --->  Receive input message end
/// Call horus_code_block_read() until done    --->
///                                            <---  Return output message begin
///                                            <---  Return output message data
///                                                    with possible buffer request
///                                            <---  ....
///                                            <---  Return output message begin
///                                            <---  Return output message data
///                                                    with possible buffer request
///                                            <---  ....
///                                            <---  Return subscriptions
///                                            <---  Return logs
/// If buffers are requested,
///   call horus_code_block_write()
///   for each buffer request                  --->  Receive and write to buffer
/// If buffers are requested,
///   call horus_code_block_read() until done  --->
///                                            <---  Return subscriptions
///                                            <---  Return logs
/// Release input message lock
/// Send output messages
///
///
/// Control flow during Horus Code Block component stop
///
/// COMPONENT                                        SHARED LIBRARY
///
/// Call horus_code_block_write()              --->  Receive stop
/// Call horus_code_block_read() until done    --->
///                                            <---  Return subscriptions
///                                            <---  Return logs
///
///
/// Control flow during Horus Code Block component destruction
///
/// COMPONENT                                        SHARED LIBRARY
///
/// Call horus_code_block_close()              --->  Destroy user context
/// Close shared library file

#ifndef HORUS_CODE_BLOCK_H
#define HORUS_CODE_BLOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/// Version of this Horus Code Block C API.
#define HORUS_CODE_BLOCK_VERSION 0u

// =================================================================================================
// -- Types ----------------------------------------------------------------------------------------
// =================================================================================================

// -- Return types ---------------------------------------------------------------------------------

/// Return type of all Horus Code Block C API functions.  This should be either
/// Horus_code_block_success (0) or one of the negative
/// Horus_code_block_error_enum values.  In case of horus_code_block_read(), a
/// successful result may be composed using the positive
/// Horus_code_block_read_flag_enum values.
typedef int Horus_code_block_result;

enum
{
    /// Return value indicating success.
    Horus_code_block_success = 0
};

/// Return values of Horus Code Block C API functions that indicate an error.
enum Horus_code_block_error_enum
{
    /// Return value indicating an error.
    Horus_code_block_error = -1
};

enum Horus_code_block_read_flag_enum
{
    Horus_code_block_read_flag_read_more = (1 << 1),
};

// -- Discovery  -----------------------------------------------------------------------------------

/// Discovery information.
struct Horus_code_block_discovery_info
{
    /// The name of a discovered Horus Code Block component.  A null-terminated
    /// string.  May be NULL.
    const char *name;

    /// The description of a discovered Horus Code Block component.  A
    /// null-terminated string.  May be NULL.
    const char *description;
};

// -- Context  -------------------------------------------------------------------------------------

/// A key-value pair.
struct Horus_code_block_key_value
{
    /// A null-terminated string.  May not be NULL.
    const char *key;

    /// A null-terminated string.  May not be NULL.
    const char *value;
};

/// An array of key-value pairs.
struct Horus_code_block_key_value_array
{
    size_t size;

    const struct Horus_code_block_key_value *array;
};

/// User defined context information specific to a Horus Code Block component
/// instance.
typedef void Horus_code_block_user_context;

/// Information on a Horus Code Block component instance.
struct Horus_code_block_context
{
    // -- User context -----------------------------------------------------------------------------

    Horus_code_block_user_context *user_context;

    // -- Component instance information -----------------------------------------------------------

    /// The ID (name) of the Horus Code Block component instance.  A
    /// null-terminated string.  May not be NULL.
    const char *component_instance_id;

    /// All key-value pairs of the "Key-Value Pairs" property of the Horus Code
    /// Block component instance.
    struct Horus_code_block_key_value_array key_value_array;
};

// -- Write and read data --------------------------------------------------------------------------

enum Horus_code_block_data_type_enum
{
    Horus_code_block_data_type_none = 0,

    // -- Logs (read only) -------------------------------------------------------------------------

    /// Horus_code_block_data_log
    Horus_code_block_data_type_log = 1,

    // -- Subscriptions (read only) ----------------------------------------------------------------

    /// Horus_code_block_data_subscriptions
    Horus_code_block_data_type_subscriptions = 2,

    // -- State transitions (write only) -----------------------------------------------------------

    Horus_code_block_data_type_start = 3,
    Horus_code_block_data_type_stop = 4,

    // -- Input messages (write only) --------------------------------------------------------------

    /// Horus_code_block_data_input_message_begin
    Horus_code_block_data_type_input_message_begin = 5,
    Horus_code_block_data_type_input_message_end = 6,

    // -- Output messages (read only) --------------------------------------------------------------

    /// Horus_code_block_data_output_message_begin
    Horus_code_block_data_type_output_message_begin = 7,

    // -- Message data (read and write) ------------------------------------------------------------

    /// Horus_code_block_data_grabber_utc_timestamp
    Horus_code_block_data_type_grabber_utc_timestamp = 8,
    /// Horus_code_block_data_ascii_in (write only)

    Horus_code_block_data_type_ascii_in = 9,
    /// Horus_code_block_data_ascii_out (read_only, with buffer request)
    Horus_code_block_data_type_ascii_out = 10,
    /// Horus_code_block_data_video_fourcc_in (write only)
    Horus_code_block_data_type_video_fourcc_in = 11,
    /// Horus_code_block_data_video_fourcc_out (read_only, with buffer request)
    Horus_code_block_data_type_video_fourcc_out = 12,

    /// Horus_code_block_data_ptz_stop
    Horus_code_block_data_type_ptz_stop = 13,
    /// Horus_code_block_data_ptz_button
    Horus_code_block_data_type_ptz_button = 14,
    /// Horus_code_block_data_ptz_orientation_rpy
    Horus_code_block_data_type_ptz_orientation_rpy = 17,

    /// Horus_code_block_data_sensor
    Horus_code_block_data_type_sensor = 15,

    /// Horus_code_block_data_buffer (write_only, reply to buffer request)
    Horus_code_block_data_type_buffer = 16,

    // Horus Nvidia / Cuda extensions
    Horus_cuda_code_block_buffer_info = 100
};
typedef int Horus_code_block_data_type;

struct Horus_code_block_data
{
    Horus_code_block_data_type type;

    const void *contents;
};

// -- Logs (read only) --

/// Severity of a log message.
enum Horus_code_block_log_severity_enum
{
    /// To log events for debugging purposes.
    Horus_code_block_log_severity_debug,
    /// To log events that are not an error.
    Horus_code_block_log_severity_info,
    /// To log events that may be an error.
    Horus_code_block_log_severity_warning,
    /// To log errors from which the application can recover.
    Horus_code_block_log_severity_error
};
typedef int Horus_code_block_log_severity;

/// Horus_code_block_data_type_log
struct Horus_code_block_data_log
{
    Horus_code_block_log_severity severity;

    /// A null-terminated string.  May not be NULL.
    const char *text;
};

// -- Subscriptions (read only) --

enum Horus_code_block_input_pipe_enum
{
    Horus_code_block_input_pipe_data_0 = 0,
    Horus_code_block_input_pipe_data_1 = 1,
    Horus_code_block_input_pipe_data_2 = 2,
    Horus_code_block_input_pipe_clock = 3
};
typedef int Horus_code_block_input_pipe;

/// Horus_code_block_data_type_subscriptions
struct Horus_code_block_data_subscriptions
{
    /// Should not be zero.
    size_t input_pipe_array_size;

    /// May not be NULL.
    const Horus_code_block_input_pipe *input_pipe_array;

    size_t input_message_data_type_subscription_array_size;

    /// Any number of message data types.  May be NULL only if
    /// input_message_data_type_subscription_array_size is zero.
    const Horus_code_block_data_type *input_message_data_type_subscription_array;
};

// -- Input messages (write only) --

/// Horus_code_block_data_type_input_message_begin
struct Horus_code_block_data_input_message_begin
{
    Horus_code_block_input_pipe input_pipe;

    /// A null-terminated string.  May not be NULL.
    const char *sender_id;

    /// A null-terminated string.  May not be NULL.
    const char *source_id;
};

// -- Output messages (read only) --

enum Horus_code_block_output_pipe_enum
{
    Horus_code_block_output_pipe_data_0 = 0,
    Horus_code_block_output_pipe_data_1 = 1,
    Horus_code_block_output_pipe_data_2 = 2,
    Horus_code_block_output_pipe_data_3 = 3
};
typedef int Horus_code_block_output_pipe;

/// Horus_code_block_data_type_output_message_begin
struct Horus_code_block_data_output_message_begin
{
    Horus_code_block_output_pipe output_pipe;
};

// -- Message data (read and write) --

typedef unsigned int Horus_code_block_request_id;

// Horus_code_block_data_type_grabber_utc_timestamp
struct Horus_code_block_data_grabber_utc_timestamp
{
    /// May be NULL.
    const bool *system_coordinate;

    /// Number of microseconds since January 1, 1970.
    uint64_t stamp;
};

/// Horus_code_block_data_type_ascii_in (write only)
struct Horus_code_block_data_ascii_in
{
    /// A null-terminated string.  May be NULL.
    const char *data_id;

    size_t buffer_size;

    /// May be NULL only if buffer_size is zero.
    const char *buffer;

    /// Ignored on read.
    size_t stream_index_array_size;

    /// May be NULL only if stream_index_array_size is zero.  Elements are
    /// null-terminated strings which may not be NULL.
    const char *const *stream_index_array;
};

/// Horus_code_block_data_type_ascii_out (read only, with buffer request)
struct Horus_code_block_data_ascii_out
{
    /// A null-terminated string.  May be NULL, in which case the name of the
    /// Horus Code Block component is used instead.
    const char *data_id;

    /// Must be unique per input message.
    Horus_code_block_request_id buffer_request_id;

    size_t buffer_size;
};

/// Horus_code_block_data_type_video_fourcc_in (write only)
struct Horus_code_block_data_video_fourcc_in
{
    /// A null-terminated string.  May be NULL.
    const char *data_id;

    size_t buffer_size;

    /// May be NULL only if buffer_size is zero.
    const char *buffer;

    size_t stream_index_array_size;

    /// May be NULL only if stream_index_array_size is zero.  Elements are
    /// null-terminated strings which may not be NULL.
    const char *const *stream_index_array;

    char fourcc[4];

    /// May be NULL.
    const uint32_t *width;

    /// May be NULL.
    const uint32_t *height;

    /// May be NULL.
    const bool *is_key_frame;

    /// May be NULL.
    const uint32_t *line_stride;
};

/// Horus_code_block_data_type_video_fourcc_out (read only, with buffer request)
struct Horus_code_block_data_video_fourcc_out
{
    /// A null-terminated string.  May be NULL, in which case the name of the
    /// Horus Code Block component is used instead.
    const char *data_id;

    /// Must be unique per input message.
    Horus_code_block_request_id buffer_request_id;

    size_t buffer_size;

    char fourcc[4];

    /// May be NULL.
    const uint32_t *width;

    /// May be NULL.
    const uint32_t *height;

    /// May be NULL.
    const bool *is_key_frame;

    /// May be NULL.
    const uint32_t *line_stride;
};

/// Horus_code_block_data_type_ptz_stop
struct Horus_code_block_data_ptz_stop
{
    /// A null-terminated string.  May be NULL.
    const char *id;

    bool stop;
};

/// Horus_code_block_data_type_ptz_button
struct Horus_code_block_data_ptz_button
{
    /// A null-terminated string.  May be NULL.
    const char *id;

    /// A null-terminated string.  May not be NULL.
    const char *button_id_pressed;
};

enum Horus_code_block_unit_enum
{
    Horus_code_block_unit_unknown_unit = 0,
    Horus_code_block_unit_unknown = 1,
    Horus_code_block_unit_degree = 2,
    Horus_code_block_unit_radian = 3,
    Horus_code_block_unit_celcius = 4,
    Horus_code_block_unit_normalized = 5,
    Horus_code_block_unit_degree_sec = 22,
    Horus_code_block_unit_radian_sec = 23,
    Horus_code_block_unit_g = 24,              ///< g-force
    Horus_code_block_unit_gauss = 25,          ///< Gauss
    Horus_code_block_unit_percentage = 26,     ///< (%) 1 part in 100
    Horus_code_block_unit_v = 27,              ///< Volts
    Horus_code_block_unit_w = 28,              ///< Watts
    Horus_code_block_unit_a = 29,              ///< Amps
    Horus_code_block_unit_b = 30,              ///< Bit (recommended by IEEE 1541 (2002)).
    Horus_code_block_unit_quaternion = 31,     ///< http://en.wikipedia.org/wiki/Quaternion
    Horus_code_block_unit_mm = 32,             ///< Millimeters
    Horus_code_block_unit_m = 33,              ///< Meters
    Horus_code_block_unit_km = 34,             ///< Kilometers
    Horus_code_block_unit_m_sec = 39,          ///< Meters per second
    Horus_code_block_unit_kmph = 40,           ///< Kilometers per hour
    Horus_code_block_unit_bpm = 41,            ///< Beats per minute (for heartrate)
    Horus_code_block_unit_px = 42,             ///< Pixel
    Horus_code_block_unit_usec = 50,           ///< Microseconds
    Horus_code_block_unit_kg = 51,             ///< kilogram
    Horus_code_block_unit_mbar = 52,           ///< millibar
    Horus_code_block_unit_rpm = 53,            ///< rotations per minute
    Horus_code_block_unit_mhz = 54,            ///< megahertz
    Horus_code_block_unit_gb = 55,             ///< gigabyte
    Horus_code_block_unit_ah = 56,             ///< Ampere hour
    Horus_code_block_unit_pa = 57,             ///< Pascal
    Horus_code_block_unit_kpa = 58,            ///< kilopascal
    Horus_code_block_unit_grams_sec = 59,      ///< grams/sec
    Horus_code_block_unit_fuel_air_ratio = 60, ///< Fuel-air equivalence ratio (used in OBD-II)
    Horus_code_block_unit_l_hour = 61,         ///< Liters per hour
    Horus_code_block_unit_sec = 62,            ///< Seconds
    Horus_code_block_unit_minute = 63          ///< Minutes
};
typedef int Horus_code_block_unit;

/// Horus_code_block_data_type_ptz_orientation_rpy
struct Horus_code_block_data_ptz_orientation_rpy
{
    /// A null-terminated string.  May be NULL.
    const char *id;

    double roll;

    /// May be NULL.
    const Horus_code_block_unit *roll_unit;

    double pitch;

    /// May be NULL.
    const Horus_code_block_unit *pitch_unit;

    double yaw;

    /// May be NULL.
    const Horus_code_block_unit *yaw_unit;
};

enum Horus_code_block_sensor_data_enum
{
    Horus_code_block_sensor_data_string = 1,
    Horus_code_block_sensor_data_integer = 2,
    Horus_code_block_sensor_data_double = 3,
    Horus_code_block_sensor_data_float = 4,
    Horus_code_block_sensor_data_long = 5,
    Horus_code_block_sensor_data_unsigned_integer = 6,
    Horus_code_block_sensor_data_unsigned_long = 7
};
typedef int Horus_code_block_sensor_data;

/// Horus_code_block_data_type_sensor
struct Horus_code_block_data_sensor
{
    /// A null-terminated string.  May be NULL.
    const char *name;

    /// May be NULL.
    const Horus_code_block_unit *unit;

    Horus_code_block_sensor_data data;

    /// For a scalar value, dimension_array_size is 0 and the xxx_value_array
    /// corresponding to data has size 1.  For a vector of values,
    /// dimension_array_size is 1 and dimension_array[0] is the size of the
    /// xxx_value_array corresponding to data.  For a 2 dimensional matrix of
    /// values, dimension_array_size is 2, dimension_array[0] is the number of
    /// rows, dimension_array[1] is the number of columns and dimension_array[0]
    /// * dimension_array[1] is size of the xxx_value_array corresponding to
    /// data.  For a n dimensional matrix of values, dimension_array_size is n,
    /// dimension_array[0] through dimension_array[n-1] are the various
    /// dimensions and the product of dimension_array[0] through
    /// dimension_array[n-1] is size of the xxx_value_array corresponding to
    /// data.
    size_t dimension_array_size;

    /// May be NULL only if dimension_array_size is zero.  Elements should not
    /// be negative.
    const int32_t *dimension_array;

    /// If data is Horus_code_block_sensor_data_string, it may be NULL only if
    /// its size (the product of the elements of dimension_array) is zero.  If
    /// data differs from Horus_code_block_sensor_data_string, it should be
    /// NULL.
    const char *const *string_value_array;

    /// If data is Horus_code_block_sensor_data_integer, it may be NULL only if
    /// its size (the product of the elements of dimension_array) is zero.  If
    /// data differs from Horus_code_block_sensor_data_integer, it should be
    /// NULL.
    const int32_t *integer_value_array;

    /// If data is Horus_code_block_sensor_data_double, it may be NULL only if
    /// its size (the product of the elements of dimension_array) is zero.  If
    /// data differs from Horus_code_block_sensor_data_double, it should be
    /// NULL.
    const double *double_value_array;

    /// If data is Horus_code_block_sensor_data_float, it may be NULL only if
    /// its size (the product of the elements of dimension_array) is zero.  If
    /// data differs from Horus_code_block_sensor_data_float, it should be NULL.
    const float *float_value_array;

    /// If data is Horus_code_block_sensor_data_long, it may be NULL only if its
    /// size (the product of the elements of dimension_array) is zero.  If data
    /// differs from Horus_code_block_sensor_data_long, it should be NULL.
    const int64_t *long_value_array;

    /// If data is Horus_code_block_sensor_data_unsigned_integer, it may be NULL
    /// only if its size (the product of the elements of dimension_array) is
    /// zero.  If data differs from
    /// Horus_code_block_sensor_data_unsigned_integer, it should be NULL.
    const uint32_t *unsigned_integer_value_array;

    /// If data is Horus_code_block_sensor_data_unsigned_long, it may be NULL
    /// only if its size (the product of the elements of dimension_array) is
    /// zero.  If data differs from Horus_code_block_sensor_data_unsigned_long,
    /// it should be NULL.
    const uint64_t *unsigned_long_value_array;

    /// For a scalar metadata, metadata_dimension_array_size is 0 and
    /// metadata_array has size 1.  For a vector of metadata,
    /// metadata_dimension_array_size is 1 and metadata_dimension_array[0] is
    /// the size of metadata_array.  For a 2 dimensional matrix of metadata,
    /// metadata_dimension_array_size is 2, metadata_dimension_array[0] is the
    /// number of rows, metadata_dimension_array[1] is the number of columns and
    /// metadata_dimension_array[0] * metadata_dimension_array[1] is size of
    /// metadata_array.  For a n metadata_dimensional matrix of metadata,
    /// metadata_dimension_array_size is n, metadata_dimension_array[0] through
    /// metadata_dimension_array[n-1] are the various metadata dimensions and
    /// the product of metadata_dimension_array[0] through
    /// metadata_dimension_array[n-1] is size of metadata_array.
    size_t metadata_dimension_array_size;

    /// May be NULL only if metadata_dimension_array_size is zero.  Elements should
    /// not be negative.
    const int32_t *metadata_dimension_array;

    /// May be NULL only if its size (the product of the elements of
    /// metadata_dimension_array) is zero.  Elements are null-terminated strings
    /// which may not be NULL.
    const char *const *metadata_array;
};

/// Horus_code_block_data_type_buffer (write_only, reply to buffer request)
struct Horus_code_block_data_buffer
{
    /// Must be unique per input message.
    Horus_code_block_request_id buffer_request_id;

    /// May not be zero.
    size_t buffer_size;

    /// May not be NULL.
    char *buffer;
};

// =================================================================================================
// -- Functions ------------------------------------------------------------------------------------
// =================================================================================================

#if defined(__cplusplus)
#define HORUS_CODE_BLOCK_EXTERN_C extern "C"
#else
#define HORUS_CODE_BLOCK_EXTERN_C
#endif

#ifdef _WIN32
#define HORUS_CODE_BLOCK_DLLEXPORT __declspec(dllexport)
#else
#define HORUS_CODE_BLOCK_DLLEXPORT
#endif

#define HORUS_CODE_BLOCK_API HORUS_CODE_BLOCK_EXTERN_C HORUS_CODE_BLOCK_DLLEXPORT

// -- Mandatory version function -------------------------------------------------------------------

/// Called when discovering or initializing a Horus Code Block component.
///
/// @param[out] version HORUS_CODE_BLOCK_VERSION.
HORUS_CODE_BLOCK_API Horus_code_block_result horus_code_block_get_version(unsigned int *version);

// -- Optional discovery function ------------------------------------------------------------------

/// Called when discovering a Horus Code Block component.
///
/// @param[out] discovery_info Information used when discovering a Horus Code
/// Block component.
HORUS_CODE_BLOCK_API Horus_code_block_result
horus_code_block_get_discovery_info(const struct Horus_code_block_discovery_info **discovery_info);

// -- Mandatory functions --------------------------------------------------------------------------

/// Called when initializing a Horus Code Block component.
///
/// Use this function to initialize the user context.
///
/// @param[in] context Information on the Horus Code Block component instance
/// calling this function.
///
/// @param[out] user_context A non-const pointer to the user_context member of
/// @a context.
HORUS_CODE_BLOCK_API Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *context,
    Horus_code_block_user_context **user_context);

/// Called before re-initializing or when destroying a Horus Code Block
/// component.
///
/// Use this function to destroy the user context.
///
/// @param[in] context Information on the Horus Code Block component instance
/// calling this function.
HORUS_CODE_BLOCK_API Horus_code_block_result
horus_code_block_close(const struct Horus_code_block_context *context);

/// Called when a Horus Code Block component needs to transfer data to its
/// shared library.
///
/// @param[in] context Information on the Horus Code Block component instance
/// calling this function.
///
/// @param[in] data The data to transfer.
HORUS_CODE_BLOCK_API
Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *context,
    const struct Horus_code_block_data *data);

/// Called when a Horus Code Block component needs to transfer data from its
/// shared library.
///
/// @param[in] context Information on the Horus Code Block component instance
/// calling this function.
///
/// @param[out] data The data to transfer.
HORUS_CODE_BLOCK_API
Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *context,
    const struct Horus_code_block_data **data);

#endif // HORUS_CODE_BLOCK_H
