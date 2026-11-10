#include "faceBlendshapes.h"

#include <cmath>

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

int subsetIndex(int landmark)
{
    for (int i = 0; i < kNumSubset; ++i)
        if (kLandmarksSubset[i] == landmark) return i;
    return -1;
}

// Measured on the canonical face: with the iris centre on the line between the
// eye corners the model reads the eyes as looking down (eyeLookDown ~0.5).
// Raising it 0.1 eye widths keeps every eyeLook* at or below 0.2. Higher turns
// the gaze up and also raises eyeBlink, which the model partly reads from the
// lids' position relative to the iris.
constexpr float kIrisRaise = 0.1f;
constexpr float kIrisRadius = 0.2f; // ~12 mm iris over a ~30 mm eye

} // namespace

void synthesizeIris(float *xy)
{
    // Eye corners in image left-to-right order for an unmirrored face. The right
    // eye's iris is 468-472, the left's 473-477.
    static const int eyes[2][2] = {{33, 133}, {362, 263}};
    const float *forehead = xy + 2 * subsetIndex(10);
    for (int e = 0; e < 2; ++e) {
        const float *a = xy + 2 * subsetIndex(eyes[e][0]);
        const float *b = xy + 2 * subsetIndex(eyes[e][1]);
        const float w = std::hypot(b[0] - a[0], b[1] - a[1]);
        const float ux = w > 0 ? (b[0] - a[0]) / w : 1.0f;
        const float uy = w > 0 ? (b[1] - a[1]) / w : 0.0f;
        const float mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        // "Up" for the eye: perpendicular to the corner line, toward the forehead.
        // Using the eye's own axes keeps this right when the head rolls in the image.
        float nx = uy, ny = -ux;
        if (nx * (forehead[0] - mx) + ny * (forehead[1] - my) < 0) {
            nx = -nx;
            ny = -ny;
        }
        const float cx = mx + kIrisRaise * w * nx, cy = my + kIrisRaise * w * ny;
        const float r = kIrisRadius * w;
        // MediaPipe's order: centre, then the ring starting along the corner line.
        const float iris[5][2] = {
            {cx, cy},
            {cx + r * ux, cy + r * uy},
            {cx + r * nx, cy + r * ny},
            {cx - r * ux, cy - r * uy},
            {cx - r * nx, cy - r * ny},
        };
        float *out = xy + 2 * subsetIndex(kNumFaceLandmarks + 5 * e);
        for (int k = 0; k < 5; ++k) {
            out[2 * k] = iris[k][0];
            out[2 * k + 1] = iris[k][1];
        }
    }
}

} // namespace fbs
