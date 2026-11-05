#include "faceBlendshapes.h"
#include "tfliteRuntime.h"

#include <cstdio>

namespace fbs {

// clang-format off
const std::array<int, kNumSubset> kLandmarksSubset = {
    0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,
    40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
    81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
};

const std::array<const char *, kNumBlendshapes> kBlendshapeNames = {
    "_neutral",
    "browDownLeft",        "browDownRight",       "browInnerUp",
    "browOuterUpLeft",     "browOuterUpRight",    "cheekPuff",
    "cheekSquintLeft",     "cheekSquintRight",    "eyeBlinkLeft",
    "eyeBlinkRight",       "eyeLookDownLeft",     "eyeLookDownRight",
    "eyeLookInLeft",       "eyeLookInRight",      "eyeLookOutLeft",
    "eyeLookOutRight",     "eyeLookUpLeft",       "eyeLookUpRight",
    "eyeSquintLeft",       "eyeSquintRight",      "eyeWideLeft",
    "eyeWideRight",        "jawForward",          "jawLeft",
    "jawOpen",             "jawRight",            "mouthClose",
    "mouthDimpleLeft",     "mouthDimpleRight",    "mouthFrownLeft",
    "mouthFrownRight",     "mouthFunnel",         "mouthLeft",
    "mouthLowerDownLeft",  "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight",     "mouthPucker",         "mouthRight",
    "mouthRollLower",      "mouthRollUpper",      "mouthShrugLower",
    "mouthShrugUpper",     "mouthSmileLeft",      "mouthSmileRight",
    "mouthStretchLeft",    "mouthStretchRight",   "mouthUpperUpLeft",
    "mouthUpperUpRight",   "noseSneerLeft",       "noseSneerRight",
};
// clang-format on

namespace {

std::array<int, kNumBlendshapes> buildMirrorTable()
{
    std::array<int, kNumBlendshapes> out{};
    auto swapSide = [](std::string n) {
        const auto ends = [&n](const std::string &s) {
            return n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0;
        };
        if (ends("Left")) return n.substr(0, n.size() - 4) + "Right";
        if (ends("Right")) return n.substr(0, n.size() - 5) + "Left";
        return n;
    };
    for (int i = 0; i < kNumBlendshapes; ++i) {
        out[i] = i;
        const std::string other = swapSide(kBlendshapeNames[i]);
        for (int j = 0; j < kNumBlendshapes; ++j)
            if (other == kBlendshapeNames[j]) out[i] = j;
    }
    return out;
}

} // namespace

// Defined after kBlendshapeNames in this file, so it is initialized after it.
const std::array<int, kNumBlendshapes> kMirroredBlendshape = buildMirrorTable();

namespace {

void captureError(void *userData, const char *format, va_list args)
{
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), format, args);
    *static_cast<std::string *>(userData) = buf;
}

} // namespace

BlendshapeModel::~BlendshapeModel() { unload(); }

void BlendshapeModel::unload()
{
    const tfl::Api *a = tfl::api();
    if (a) {
        if (m_interpreter) a->TfLiteInterpreterDelete(m_interpreter);
        if (m_model) a->TfLiteModelDelete(m_model);
    }
    m_interpreter = nullptr;
    m_model = nullptr;
    m_path.clear();
}

bool BlendshapeModel::load(const std::string &modelPath, std::string &error)
{
    unload();
    const tfl::Api *a = tfl::api();
    if (!a) {
        error = "TensorFlow Lite library is not loaded";
        return false;
    }

    m_model = a->TfLiteModelCreateFromFile(modelPath.c_str());
    if (!m_model) {
        error = "Could not read TFLite model: " + modelPath;
        return false;
    }

    tfl::TfLiteInterpreterOptions *opts = a->TfLiteInterpreterOptionsCreate();
    // The model is tiny (~1 MB MLP-Mixer). One thread avoids fighting Maya's
    // parallel evaluation for cores.
    a->TfLiteInterpreterOptionsSetNumThreads(opts, 1);
    a->TfLiteInterpreterOptionsSetErrorReporter(opts, captureError, &m_lastTfError);
    m_interpreter = a->TfLiteInterpreterCreate(m_model, opts);
    a->TfLiteInterpreterOptionsDelete(opts);

    auto fail = [&](const std::string &msg) {
        error = msg + (m_lastTfError.empty() ? "" : (": " + m_lastTfError));
        unload();
        return false;
    };

    if (!m_interpreter) return fail("Could not create TFLite interpreter for " + modelPath);
    if (a->TfLiteInterpreterAllocateTensors(m_interpreter) != tfl::kTfLiteOk)
        return fail("Could not allocate TFLite tensors");

    if (a->TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1 ||
        a->TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1)
        return fail("Unexpected tensor count; is this face_blendshapes.tflite?");

    const tfl::TfLiteTensor *in = a->TfLiteInterpreterGetInputTensor(m_interpreter, 0);
    const tfl::TfLiteTensor *out = a->TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
    const bool inOk = a->TfLiteTensorType(in) == tfl::kTfLiteFloat32 &&
                      a->TfLiteTensorByteSize(in) == sizeof(float) * kNumSubset * 2;
    const bool outOk = a->TfLiteTensorType(out) == tfl::kTfLiteFloat32 &&
                       a->TfLiteTensorByteSize(out) == sizeof(float) * kNumBlendshapes;
    if (!inOk || !outOk)
        return fail("Unexpected tensor shapes; expected float32 [1,146,2] -> [52]. "
                    "Is this face_blendshapes.tflite?");

    m_path = modelPath;
    return true;
}

bool BlendshapeModel::run(const float *xy, float *scores, std::string &error)
{
    const tfl::Api *a = tfl::api();
    if (!a || !m_interpreter) {
        error = "Model is not loaded";
        return false;
    }
    m_lastTfError.clear();
    tfl::TfLiteTensor *in = a->TfLiteInterpreterGetInputTensor(m_interpreter, 0);
    if (a->TfLiteTensorCopyFromBuffer(in, xy, sizeof(float) * kNumSubset * 2) != tfl::kTfLiteOk ||
        a->TfLiteInterpreterInvoke(m_interpreter) != tfl::kTfLiteOk) {
        error = "TFLite inference failed: " + m_lastTfError;
        return false;
    }
    const tfl::TfLiteTensor *out = a->TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
    if (a->TfLiteTensorCopyToBuffer(out, scores, sizeof(float) * kNumBlendshapes) !=
        tfl::kTfLiteOk) {
        error = "Could not read TFLite output: " + m_lastTfError;
        return false;
    }
    return true;
}

} // namespace fbs
