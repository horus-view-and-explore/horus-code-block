// This file contains CPU Temperature, a C++ example implementation of the Horus
// Code Block C API.
//
// Copyright (C) 2020, 2021, 2024 Horus View and Explore B.V.
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

/*
 @author  Auke-Dirk Pietersma
 @version 1.0, 15/01/2025
 @description Integration of TensorRT within the Horison
*/

#include "../../../Horus_code_block.h"      // codec block interface
#include "../../../Horus_cuda_code_block.h" // cuda codec block interface
#include "../Helper_functions.hpp"
#include "../Horus_cuda.hpp" // Cuda buffers implementation
#include "Kernel.h"          // Cuda kernels
#include "Network.hpp"       // Network (TensorRT) definition
#include "Transforms.hpp"    // Generate mapping from network tensor to image

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

// The Main User context
struct User_context;

// Initialize the tensorflow network
void try_init_tensor_rt(User_context *ctx, Horus_cuda_code_block_data_buffer *input);
void init_tensor_rt(User_context *ctx);
void check_tensor_rt(User_context *ctx);
void tensor_rt_destroy(User_context *ctx);
bool generate_buffers(User_context *ctx);

// The three main actions
void run(User_context *ctx);
void convert_input(User_context *ctx);
void convert_output(User_context *ctx);

// =================================================================================================
// -- Context data --------------------------------------------------------------------------------
// =================================================================================================

struct User_context
{
    std::string path;
    std::string debug_path;
    std::string testfile;

    // Our original input buffer from system (Horison)
    Horus_cuda_code_block_data_buffer *input;

    //(input) In order of transformation
    std::unique_ptr<hrs::HC_buffer> hcb_input;
    std::unique_ptr<hrs::HC_buffer> hcb_input_scaled;
    std::unique_ptr<hrs::HC_buffer> hcb_input_planar;
    std::unique_ptr<hrs::HC_buffer> hcb_input_float;

    //(output) In order of transformation
    std::unique_ptr<hrs::HC_buffer> hcb_output_float;
    std::unique_ptr<hrs::HC_buffer> hcb_output_char;
    std::unique_ptr<hrs::HC_buffer> hcb_output_rearange;
    std::unique_ptr<hrs::HC_buffer> hcb_output_mapping;

    // binding of buffers to the TensorRT network
    void *buffers[2];

    // The actual TensorRT network
    hrs::Network network;

    bool init;
    bool input_claimed;
    bool download;
};

void try_init_tensor_rt(User_context *ctx, Horus_cuda_code_block_data_buffer *input)
{
    Helper_functions_show_buffer_info(input);

    switch (input->index)
    {
        case 0:
            ctx->input = input;
            ctx->hcb_input = std::make_unique<hrs::HC_buffer>(input);
            std::cout << ctx->hcb_input->info() << std::endl;
            break;
    }

    if (ctx->input != nullptr)
    {
        init_tensor_rt(ctx);
    }
}

void run(User_context *ctx)
{
    convert_input(ctx);

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "pre runn");

#if NV_TENSORRT_MAJOR >= 10
    bool status = ctx->network.context->enqueueV3(ctx->input->cuda_stream);

#else
    bool status = ctx->network.context->enqueueV2(ctx->buffers, ctx->input->cuda_stream, nullptr);
#endif

    if (!status)
    {
        Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "runn error");
        tensor_rt_destroy(ctx);
    }

    if (!status)
    {
        exit(1);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "post runn");

    convert_output(ctx);
}

void convert_input(User_context *ctx)
{
    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "convert_input");
    if (ctx->download)
    {
        ctx->hcb_input->download(ctx->input->cuda_stream, true);
        ctx->hcb_input->save(ctx->debug_path);
    }

    launch_scale_image(
        (unsigned char *)ctx->hcb_input->dev,
        (unsigned char *)ctx->hcb_input_scaled->dev,
        ctx->hcb_input->width,
        ctx->hcb_input->height,
        ctx->hcb_input->channels,
        ctx->hcb_input_scaled->width,
        ctx->hcb_input_scaled->height,
        ctx->hcb_input_scaled->channels,
        ctx->input->cuda_stream);

    if (ctx->download)
    {
        ctx->hcb_input_scaled->download(ctx->input->cuda_stream, true);
        ctx->hcb_input_scaled->save(ctx->debug_path);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "launch_scale_image");

    launch_packed_to_planar(
        (unsigned char *)ctx->hcb_input_scaled->dev,
        (unsigned char *)ctx->hcb_input_planar->dev,
        ctx->hcb_input_scaled->width,
        ctx->hcb_input_scaled->height,
        ctx->input->cuda_stream);

    if (ctx->download)
    {
        ctx->hcb_input_planar->download(ctx->input->cuda_stream, true);
        ctx->hcb_input_planar->save(ctx->debug_path);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "launch_packed_to_planar");

    launch_conversion(
        (unsigned char *)ctx->hcb_input_planar->dev,
        (float *)ctx->hcb_input_float->dev,
        ctx->hcb_input_scaled->width,
        ctx->hcb_input_scaled->height,
        ctx->hcb_input_scaled->width,
        ctx->input->cuda_stream,
        true);

    if (ctx->download)
    {
        ctx->hcb_input_float->download(ctx->input->cuda_stream, true);
        ctx->hcb_input_float->save(ctx->debug_path);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "launch_char_2_float");
}

void convert_output(User_context *ctx)
{
    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "convert_ouput");

    if (ctx->download)
    {
        ctx->hcb_output_float->download(ctx->input->cuda_stream, true);
        ctx->hcb_output_float->save(ctx->debug_path);
    }

    // Convert back to the scaled RGB buffer
    launch_conversion(
        (unsigned char *)ctx->hcb_output_char->dev,
        (float *)ctx->hcb_output_float->dev,
        ctx->hcb_output_float->width,
        ctx->hcb_output_float->height,
        ctx->hcb_output_float->width,
        ctx->input->cuda_stream,
        false);

    if (ctx->download)
    {
        ctx->hcb_output_char->download(ctx->input->cuda_stream, true);
        ctx->hcb_output_char->save(ctx->debug_path);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "float->char");

    launch_rearange(
        (unsigned char *)ctx->hcb_output_char->dev,
        (unsigned char *)ctx->hcb_output_rearange->dev,
        (unsigned int *)ctx->hcb_output_mapping->dev,
        ctx->hcb_output_char->width,
        ctx->hcb_output_char->height,
        ctx->input->cuda_stream);

    if (ctx->download)
    {
        ctx->hcb_output_rearange->download(ctx->input->cuda_stream, true);
        ctx->hcb_output_rearange->save(ctx->debug_path);
    }

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "rearange");

    launch_overlay_image(
        (unsigned char *)ctx->hcb_output_rearange->dev,
        (unsigned char *)ctx->input->mem_dev,
        ctx->hcb_output_rearange->width,
        ctx->hcb_output_rearange->height,
        ctx->hcb_output_rearange->channels,
        ctx->input->media_width,
        ctx->input->media_height,
        ctx->hcb_input->channels,
        ctx->input->cuda_stream);

    Horus_cuda_code_block_synchronize(ctx->input->cuda_stream, "overlay");

    if (ctx->download)
    {
        ctx->hcb_input->download(ctx->input->cuda_stream, true);
        ctx->hcb_input->save(ctx->debug_path);
    }
}

bool loadTensorRTModel(User_context *ctx, const std::string &modelPath, int batchSize)
{
    hrs::TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(hrs::logger));
    if (!runtime)
    {
        return false;
    }

    std::ifstream modelFile(modelPath, std::ios::binary | std::ios::ate);
    if (!modelFile.good())
    {
        return false;
    }

    size_t size = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> serializedEngine(size);
    modelFile.read(serializedEngine.data(), size);
    modelFile.close();

    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(serializedEngine.data(), size);
    if (!engine)
    {
        return false;
    }

    hrs::TRTUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // ctx->network.serialized_engine = std::move(serializedEngine);
    ctx->network.engine = engine;
    ctx->network.runtime = std::move(runtime);
    ctx->network.context = std::move(context);

    return true;
}

bool loadOnnxModel(User_context *ctx, const std::string &modelPath, int batchSize)
{

    // Create builder
    hrs::TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(hrs::logger));
    if (!builder)
        return false;

    // Create network definition
    const auto explicitBatch =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    hrs::TRTUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(explicitBatch));
    if (!network)
        return false;

    // Create ONNX parser
    hrs::TRTUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, hrs::logger));
    if (!parser)
        return false;

    // Parse ONNX file
    if (!parser->parseFromFile(
            modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        return false;
    }

    // Create optimization config
    hrs::TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
        return false;

    // Set max workspace size (1GB)
    // config->setMaxWorkspaceSize(1ULL << 30);

    // Build optimized engine
    hrs::TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(
        builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine)
        return false;

    // Create runtime and deserialize engine
    hrs::TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(hrs::logger));
    if (!runtime)
        return false;

    nvinfer1::ICudaEngine *engine =
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine)
        return false;

    // Create execution context
    hrs::TRTUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context)
        return false;

    // Engine is ready for inference
    ctx->network.builder = std::move(builder);
    ctx->network.network = std::move(network);
    ctx->network.config = std::move(config);
    ctx->network.serialized_engine = std::move(serializedEngine);
    ctx->network.engine = engine;
    ctx->network.runtime = std::move(runtime);
    ctx->network.context = std::move(context);

    return true;
}

bool generate_buffers(User_context *ctx)
{
    std::string path = ctx->debug_path + ctx->testfile;

    // ****************** INPUT *******************
    size_t width = 800;
    size_t height = 800;

    // Step 1: The model requires 800x800 resolution of char (1) in RGB format (3)
    {
        ctx->hcb_input_scaled = std::make_unique<hrs::HC_buffer>(
            "input_scaled", 1, 3, width, height, "RGB", "RAWVIDEO");
        std::cout << ctx->hcb_input_scaled->info() << std::endl;
        if (!ctx->hcb_input_scaled->initialized())
        {
            exit(1);
        }
    }

    // Incomming buffer is RGB linear memory packed
    // Step 2: needs planar format:
    {
        ctx->hcb_input_planar = std::make_unique<hrs::HC_buffer>(
            "input_planar", 1, 3, width, height, "RGB", "RAWVIDEO");
        std::cout << ctx->hcb_input_planar->info() << std::endl;
        if (!ctx->hcb_input_planar->initialized())
        {
            exit(1);
        }
    }

    // Step 3: TensorRT requires float inputs
    {
        ctx->hcb_input_float = std::make_unique<hrs::HC_buffer>(
            "input_floats", sizeof(float), 3, width, height, "RGB", "RAWVIDEO");

        if (!ctx->hcb_input_float->initialized())
        {
            exit(1);
        }
    }

    // Step 4: TensorRT requires float outputs
    {
        ctx->hcb_output_float = std::make_unique<hrs::HC_buffer>(
            "output_floats", sizeof(float), 19, 400, 400, "TENSOR_OUT", "RAWVIDEO");

        if (!ctx->hcb_input_float->initialized())
        {
            exit(1);
        }
    }
    // Step 5: transform float to char
    {
        ctx->hcb_output_char = std::make_unique<hrs::HC_buffer>(
            "output_char", sizeof(char), 4, 400, 400, "TENSOR_OUT", "RAWVIDEO");

        if (!ctx->hcb_output_char->initialized())
        {
            exit(1);
        }
    }
    // Step 6: rearange the pixels to a standard image format
    {
        ctx->hcb_output_rearange = std::make_unique<hrs::HC_buffer>(
            "output_rearange", sizeof(char), 4, 400, 400, "RGBA", "RAWVIDEO");
        if (!ctx->hcb_output_rearange->initialized())
        {
            exit(1);
        }
    }

    // Step 7: create the rearange mapping
    {
        const size_t width = 400;
        const size_t height = 400;
        const std::array<size_t, 4> dimensions = {4, 4, 100, 100}; // {19, 4, 4, 100, 100}
        const std::array<size_t, 4> permutation = {2, 0, 3, 1};    // {0, 3, 1, 4, 2} 5 >> 4 dims
        auto mapping =
            generate_transpose_mapping<unsigned int>(width, height, dimensions, permutation);

        ctx->hcb_output_mapping = std::make_unique<hrs::HC_buffer>(
            "output_rearange", sizeof(unsigned int), 1, 400, 400, "MAPPING", "RAWVIDEO");

        if (!ctx->hcb_output_mapping->initialized())
        {
            exit(1);
        }

        Horus_cuda_code_block_memcpy_async(
            ctx->hcb_output_mapping->dev,
            mapping.data(),
            mapping.size() * sizeof(unsigned int),
            ctx->input->cuda_stream,
            cudaMemcpyHostToDevice);
    }

    // Bind the correct buffers to the network

    ctx->buffers[0] = ctx->hcb_input_float->dev;
    ctx->buffers[1] = ctx->hcb_output_float->dev;

#if NV_TENSORRT_MAJOR >= 10

    ctx->network.input_tensor = ctx->network.network->getInput(0);
    ctx->network.output_tensor = ctx->network.network->getOutput(0);

    const char *inputName = ctx->network.input_tensor->getName();
    const char *outputName = ctx->network.output_tensor->getName();

    ctx->network.context->setInputTensorAddress(inputName, ctx->buffers[0]);
    ctx->network.context->setOutputTensorAddress(outputName, ctx->buffers[1]);

#endif
    return true;
}

void init_tensor_rt(User_context *ctx)
{
    std::cout << " *** init_tensor_rt *** " << std::endl;

    std::filesystem::path modelPath(ctx->path);
    if (modelPath.extension() == ".onnx")
    {

        if (!loadOnnxModel(ctx, ctx->path, 1))
        {
            std::cout << "could not load onnx model" << std::endl;
            exit(1);
        }
    }
    else if (modelPath.extension() == ".engine")
    {
        if (!loadTensorRTModel(ctx, ctx->path, 1))
        {
            std::cout << "could not load TensorRT model" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cout << "unsupported model type: " << modelPath.extension() << std::endl;
    }

    if (!generate_buffers(ctx))
    {
        std::cout << "could not load onnx model" << std::endl;
        exit(1);
    }

    std::cout << "\n\n onnx loaded!! \n\n";
    ctx->init = true;
}

void tensor_rt_destroy(User_context *ctx)
{
    ctx->init = false;
    ctx->input = nullptr;

    ctx->network.builder.reset();
    ctx->network.network.reset();

    ctx->network.config.reset();
    ctx->network.serialized_engine.reset();

    ctx->network.runtime.reset();
    ctx->network.context.reset();

    ctx->init = false;
    ctx->input_claimed = false;
}

// -- Mandatory version function
// -------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_version(unsigned int *const version)
{
    *version = HORUS_CODE_BLOCK_VERSION;
    return Horus_code_block_success;
}

// -- Optional discovery function
// ------------------------------------------------------------------

Horus_code_block_result horus_code_block_get_discovery_info(
    const struct Horus_code_block_discovery_info **const discovery_info)
{
    static const std::string static_discovery_info_description = "Cuda TensorRT\n";
    static const struct Horus_code_block_discovery_info static_discovery_info
    {
        "Cuda TensorRT: ", static_discovery_info_description.c_str()
    };
    *discovery_info = &static_discovery_info;
    return Horus_code_block_success;
}

// -- Key-value pairs ------------------------------------------------------------------------------

static const char *find_value(
    const struct Horus_code_block_key_value_array *const key_value_array,
    const char *const key)
{
    for (size_t i = 0u; i < key_value_array->size; ++i)
    {
        const struct Horus_code_block_key_value key_value = key_value_array->array[i];
        if (strcmp(key_value.key, key) == 0)
        {
            return key_value.value;
        }
    }
    return NULL;
}

// -- Mandatory functions
// --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const code_block_context,
    Horus_code_block_user_context **const user_context)
{
    *user_context = new User_context;
    User_context *ctx = *(reinterpret_cast<User_context **const>(user_context));

    const auto path = find_value(&code_block_context->key_value_array, "path");
    if (path)
    {
        ctx->path = path;
    }
    else
    {
        std::cout << "\n ERROR [Cuda_tensor_rt_segmentation: No path provided for the model!]\n"
                  << std::endl;
        return Horus_code_block_error;
    }

    const auto debug_path = find_value(&code_block_context->key_value_array, "debug-path");
    if (debug_path)
    {
        ctx->download = true;
        ctx->debug_path = debug_path;
    }
    else
    {
        ctx->download = false;
    }

    ctx->input = nullptr;
    ctx->init = false;
    ctx->input_claimed = false;

    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    tensor_rt_destroy(static_cast<User_context *>(context->user_context));
    delete static_cast<User_context *>(context->user_context); // Clean up
    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_write(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data *const data)
{
    User_context *user_ctx = static_cast<User_context *>(context->user_context);

    if (data->type == Horus_cuda_code_block_buffer_info)
    {
        Horus_cuda_code_block_data_buffer *input =
            (Horus_cuda_code_block_data_buffer *)data->contents;

        switch (input->event)
        {
            case Horus_cuda_code_block_data_buffer::Horus_cuda_code_block_data_buffer_event_stopped:
            {
                if (input->index == 0)
                {
                    tensor_rt_destroy(user_ctx);
                }
                break;
            }

            case Horus_cuda_code_block_data_buffer::
                Horus_cuda_code_block_data_buffer_event_initialized:
            {
                try_init_tensor_rt(user_ctx, input);
                break;
            }

            case Horus_cuda_code_block_data_buffer::
                Horus_cuda_code_block_data_buffer_event_slot_changed:
            {
                // if my slot_idx is the active slot, then i can use the buffer
                if (input->slot_idx == input->active_slot_idx)
                {
                    if (input->index == 0)
                    {
                        user_ctx->input_claimed = true;
                    }
                }

                if (user_ctx->input_claimed)
                {
                    if (user_ctx->init)
                    {
                        run(user_ctx);

                        // Signal the cuda code block where er done.
                        input->notify_done_flags = Horus_cuda_code_block_data_buffer::
                            Notify_done_flags::Horus_cuda_code_block_data_buffer_notify_done_0;

                        // internal bookkeeping
                        user_ctx->input_claimed = false;

                        // ignore further downloading of horus cuda buffers
                        user_ctx->download = false;
                    }
                }
                break;
            }

            default:
                break;
        }
    }

    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_read(
    const struct Horus_code_block_context *const context,
    const struct Horus_code_block_data **const data)
{
    return Horus_code_block_success;
}
