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

#include "../../../Horus_code_block.h"
#include "../../../Horus_cuda_code_block.h"
#include "../Helper_functions.hpp"
#include "../Horus_cuda.hpp" // Cuda buffers implementation
#include "Network.hpp"

#include <cmath>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

void launchConversion(
    unsigned char *char_buffer,
    float *float_buffer,
    int width,
    int height,
    int stride,
    cudaStream_t stream,
    bool toFloat = true);

// The Main User context
struct User_context;

// Initialize the tensorflow network / execture
void try_init_tensor_rt(User_context *ctx, Horus_cuda_code_block_data_buffer *input);
void init_tensor_rt(User_context *ctx);
void tensor_rt_destroy(User_context *ctx);
void convert_input(User_context *ctx);
void convert_ouput(User_context *ctx);
void check_tensor_rt(User_context *ctx);
void generate_buffers(User_context *ctx);
void run(User_context *ctx, Horus_cuda_code_block_data_buffer *buffer);

struct User_context
{
    Horus_cuda_code_block_data_buffer *input;
    Horus_cuda_code_block_data_buffer *output;

    std::unique_ptr<hrs::HC_buffer> hcb_input;
    std::unique_ptr<hrs::HC_buffer> hcb_output;
    std::unique_ptr<hrs::HC_buffer> hcb_input_float;
    std::unique_ptr<hrs::HC_buffer> hcb_output_float;

    hrs::Network network;

    bool init;
    bool input_claimed;
    bool output_claimed;

    void *buffers[2];
};

void generate_buffers(User_context *ctx)
{
    // ** Original input / Output buffers wrapped in HC_buffer
    ctx->hcb_input = std::make_unique<hrs::HC_buffer>(ctx->input);
    std::cout << ctx->hcb_input->info() << std::endl;

    ctx->hcb_output = std::make_unique<hrs::HC_buffer>(ctx->output);
    std::cout << ctx->hcb_output->info() << std::endl;

    // Create the float buffers
    ctx->hcb_input_float = std::make_unique<hrs::HC_buffer>(
        "input_floats",
        sizeof(float),
        4,
        ctx->hcb_input->width,
        ctx->hcb_input->height,
        "RGBA",
        "RAWVIDEO");
    std::cout << ctx->hcb_input_float->info() << std::endl;

    ctx->hcb_output_float = std::make_unique<hrs::HC_buffer>(
        "output_floats",
        sizeof(float),
        4,
        ctx->hcb_input->width,
        ctx->hcb_input->height,
        "RGBA",
        "RAWVIDEO");
    std::cout << ctx->hcb_output_float->info() << std::endl;

    // Bind the buffers
    ctx->buffers[0] = ctx->hcb_input_float->dev;
    ctx->buffers[1] = ctx->hcb_output_float->dev;
}

void try_init_tensor_rt(User_context *ctx, Horus_cuda_code_block_data_buffer *input)
{
    Helper_functions_show_buffer_info(input);
    switch (input->index)
    {
        case 0:
            ctx->input = input;
            break;
        case 1:
            ctx->output = input;
            break;
    }

    if (ctx->input != nullptr && ctx->output != nullptr)
    {
        init_tensor_rt(ctx);
    }
}

void convert_input(User_context *ctx)
{
    launchConversion(
        (unsigned char *)ctx->hcb_input->dev,
        (float *)ctx->hcb_input_float->dev,
        ctx->input->media_width,
        ctx->input->media_height,
        ctx->input->media_width,
        ctx->input->cuda_stream,
        true);
}

void convert_ouput(User_context *ctx)
{
    launchConversion(
        (unsigned char *)ctx->output->mem_dev,
        (float *)ctx->hcb_output_float->dev,
        ctx->input->media_width,
        ctx->input->media_height,
        ctx->input->media_width,
        ctx->input->cuda_stream,
        false);
}

void run(User_context *ctx, Horus_cuda_code_block_data_buffer *buffer)
{
    convert_input(ctx);

    // Check CUDA errors before execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error before execute: " << cudaGetErrorString(error) << std::endl;
    }

#if NV_TENSORRT_MAJOR >= 10
    bool status = ctx->network.context->enqueueV3(ctx->input->cuda_stream);

#else
    bool status = ctx->network.context->enqueueV2(ctx->buffers, ctx->input->cuda_stream, nullptr);
#endif

    if (!status)
    {
        std::cout << "invalid TensorRT status: " << status << std::endl;
        exit(1);
    }

    //(ctx->input->cuda_stream);

    convert_ouput(ctx);

    // Both buffer can be released
    ctx->input_claimed = false;
    ctx->output_claimed = false;

    buffer->notify_done_flags = Horus_cuda_code_block_data_buffer::Notify_done_flags::
                                    Horus_cuda_code_block_data_buffer_notify_done_0 |
                                Horus_cuda_code_block_data_buffer::Notify_done_flags::
                                    Horus_cuda_code_block_data_buffer_notify_done_1;

    if (!status)
    {
        std::cout << "\n\n ** " << status << "! \n\n **" << std::endl;
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cout << "CUDA error before execute: " << cudaGetErrorString(error) << std::endl;
        }
        tensor_rt_destroy(ctx);
    }
}

void init_tensor_rt(User_context *ctx)
{
    generate_buffers(ctx);

    { // Create builder and network
        hrs::TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(hrs::logger));

        uint32_t flag =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        hrs::TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

        ctx->network.network = std::move(network);
        ctx->network.builder = std::move(builder);
    }

    nvinfer1::Dims inputDims{
        4, {1, 4, (int)ctx->input->media_height, (int)ctx->input->media_width}};

    ctx->network.input_tensor =
        ctx->network.network->addInput("input", nvinfer1::DataType::kFLOAT, inputDims);

    float scale_value = 0.2f;
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, &scale_value, 1};
    float shift_value = 0.0f; // single bias value
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, &shift_value, 1};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    ctx->network.scale_layer = ctx->network.network->addScale(
        *ctx->network.input_tensor,
        nvinfer1::ScaleMode::kUNIFORM,
        shift, // bias
        scale, // scale
        power  // power
    );

    // Mark output
    ctx->network.output_tensor = ctx->network.scale_layer->getOutput(0);
    ctx->network.output_tensor->setType(nvinfer1::DataType::kFLOAT);
    ctx->network.network->markOutput(*ctx->network.output_tensor);

    // Create optimization config
    {
        hrs::TRTUniquePtr<nvinfer1::IBuilderConfig> config(
            ctx->network.builder->createBuilderConfig());
        ctx->network.config = std::move(config);
    }

    {
        // Build the engine
        hrs::TRTUniquePtr<nvinfer1::IHostMemory> serializedModel(
            ctx->network.builder->buildSerializedNetwork(
                *ctx->network.network, *ctx->network.config));
        if (!serializedModel)
        {
            std::cerr << "Failed to build serialized engine!" << std::endl;
            exit(1);
        }
        ctx->network.serialized_model = std::move(serializedModel);
    }

    {
        // Create runtime and engine
        hrs::TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(hrs::logger));
        hrs::TRTUniquePtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
            ctx->network.serialized_model->data(), ctx->network.serialized_model->size()));
        if (!engine)
        {
            std::cerr << "Failed to create engine!" << std::endl;
            exit(1);
        }
        ctx->network.runtime = std::move(runtime);
        ctx->network.engine = std::move(engine);
    }

    {
        hrs::TRTUniquePtr<nvinfer1::IExecutionContext> context(
            ctx->network.engine->createExecutionContext());
        ctx->network.context = std::move(context);
    }

#if NV_TENSORRT_MAJOR >= 10

    ctx->network.input_tensor = ctx->network.network->getInput(0);
    ctx->network.output_tensor = ctx->network.network->getOutput(0);

    const char *inputName = ctx->network.input_tensor->getName();
    const char *outputName = ctx->network.output_tensor->getName();

    ctx->network.context->setInputTensorAddress(inputName, ctx->buffers[0]);
    ctx->network.context->setOutputTensorAddress(outputName, ctx->buffers[1]);

#endif

    ctx->init = true;
}

void tensor_rt_destroy(User_context *ctx)
{
    ctx->init = false;
    ctx->input = nullptr;
    ctx->output = nullptr;

    ctx->network.builder.reset();
    ctx->network.network.reset();

    ctx->network.config.reset();
    ctx->network.serialized_model.reset();

    ctx->network.runtime.reset();
    ctx->network.context.reset();
    ctx->network.engine.reset();

    ctx->init = false;
    ctx->input_claimed = false;
    ctx->output_claimed = false;
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

// -- Mandatory functions
// --------------------------------------------------------------------------

Horus_code_block_result horus_code_block_open(
    const struct Horus_code_block_context *const,
    Horus_code_block_user_context **const user_context)
{
    *user_context = new User_context;
    User_context *ctx = *(reinterpret_cast<User_context **const>(user_context));
    ctx->input = nullptr;
    ctx->output = nullptr;
    ctx->init = false;
    ctx->input_claimed = false;
    ctx->output_claimed = false;

    return Horus_code_block_success;
}

Horus_code_block_result horus_code_block_close(const struct Horus_code_block_context *const context)
{
    // tensor_rt_destroy(static_cast<User_context *>(context->user_context));
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
                    if (input->index == 1)
                    {
                        user_ctx->output_claimed = true;
                    }
                }

                if (user_ctx->input_claimed && user_ctx->output_claimed)
                {
                    if (user_ctx->init)
                    {
                        run(user_ctx, input);
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