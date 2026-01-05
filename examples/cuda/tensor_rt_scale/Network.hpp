#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <memory>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace hrs {

class Logger : public nvinfer1::ILogger
{
  public:
    Severity level = Severity::kINFO;
    void log(Severity severity, const char *msg) noexcept override
    {
        if ((int)severity <= (int)level)
            std::cout << "Logger: " << (int)severity << " " << msg << std::endl;
    }
} logger;

#if NV_TENSORRT_MAJOR < 10

template <typename T>
struct TRTDestroy
{
    void operator()(T *obj) const
    {
        if (obj)
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            obj->destroy();
#pragma GCC diagnostic pop
        }
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy<T>>;
#else
template <typename T>
using TRTUniquePtr = std::unique_ptr<T>;

#endif

struct Network
{
    TRTUniquePtr<nvinfer1::IBuilder> builder;
    TRTUniquePtr<nvinfer1::INetworkDefinition> network;

    nvinfer1::ITensor *input_tensor;
    nvinfer1::ITensor *output_tensor;

    nvinfer1::IScaleLayer *scale_layer;

    TRTUniquePtr<nvinfer1::IBuilderConfig> config;
    TRTUniquePtr<nvinfer1::IHostMemory> serialized_model;

    TRTUniquePtr<nvinfer1::IRuntime> runtime;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
};

} // namespace hrs

#endif // NETWORK_HPP