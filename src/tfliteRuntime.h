#pragma once

// Minimal runtime binding to the TensorFlow Lite / LiteRT C API.
//
// The library is loaded with LoadLibrary/dlopen instead of being linked, so the
// plugin builds without TFLite headers or import libraries. Any library that
// exports the stable TFLite C API works:
//   - libLiteRt.dll           (Google's prebuilt, shipped in the ai-edge-litert wheel)
//   - tensorflowlite_c.dll    (built from tensorflow/lite/c with CMake or Bazel)
//   - libtensorflowlite_c.so / .dylib
//
// Only the handful of functions this project needs are bound. The declarations
// below mirror tensorflow/lite/core/c/c_api.h, which is ABI stable.

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tfl {

struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteTensor;

// TfLiteStatus and TfLiteType are C enums, which are int-sized on every
// platform Maya supports.
using TfLiteStatus = int;
using TfLiteType = int;
constexpr TfLiteStatus kTfLiteOk = 0;
constexpr TfLiteType kTfLiteFloat32 = 1;

using ErrorReporterFn = void (*)(void *userData, const char *format, va_list args);

struct Api {
    const char *(*TfLiteVersion)();

    TfLiteModel *(*TfLiteModelCreateFromFile)(const char *modelPath);
    void (*TfLiteModelDelete)(TfLiteModel *model);

    TfLiteInterpreterOptions *(*TfLiteInterpreterOptionsCreate)();
    void (*TfLiteInterpreterOptionsDelete)(TfLiteInterpreterOptions *options);
    void (*TfLiteInterpreterOptionsSetNumThreads)(TfLiteInterpreterOptions *options, int32_t n);
    void (*TfLiteInterpreterOptionsSetErrorReporter)(
        TfLiteInterpreterOptions *options, ErrorReporterFn reporter, void *userData
    );

    TfLiteInterpreter *(*TfLiteInterpreterCreate)(
        const TfLiteModel *model, const TfLiteInterpreterOptions *options
    );
    void (*TfLiteInterpreterDelete)(TfLiteInterpreter *interpreter);
    TfLiteStatus (*TfLiteInterpreterAllocateTensors)(TfLiteInterpreter *interpreter);
    TfLiteStatus (*TfLiteInterpreterInvoke)(TfLiteInterpreter *interpreter);
    int32_t (*TfLiteInterpreterGetInputTensorCount)(const TfLiteInterpreter *interpreter);
    int32_t (*TfLiteInterpreterGetOutputTensorCount)(const TfLiteInterpreter *interpreter);
    TfLiteTensor *(*TfLiteInterpreterGetInputTensor)(
        const TfLiteInterpreter *interpreter, int32_t index
    );
    const TfLiteTensor *(*TfLiteInterpreterGetOutputTensor)(
        const TfLiteInterpreter *interpreter, int32_t index
    );

    TfLiteType (*TfLiteTensorType)(const TfLiteTensor *tensor);
    int32_t (*TfLiteTensorNumDims)(const TfLiteTensor *tensor);
    int32_t (*TfLiteTensorDim)(const TfLiteTensor *tensor, int32_t dimIndex);
    size_t (*TfLiteTensorByteSize)(const TfLiteTensor *tensor);
    TfLiteStatus (*TfLiteTensorCopyFromBuffer)(
        TfLiteTensor *tensor, const void *inputData, size_t inputDataSize
    );
    TfLiteStatus (*TfLiteTensorCopyToBuffer)(
        const TfLiteTensor *tensor, void *outputData, size_t outputDataSize
    );
};

// Library file names to try, in order, when no explicit path is given.
std::vector<std::string> defaultLibraryNames();

// Try each candidate (a full path or a bare file name) until one loads and
// exports every required symbol. On success, `loadedFrom` names the library.
// On failure, `error` describes every attempt.
bool loadLibrary(
    const std::vector<std::string> &candidates, std::string &loadedFrom, std::string &error
);

void unloadLibrary();

// Null until loadLibrary succeeds.
const Api *api();

} // namespace tfl
