// Standalone (no Maya) check that the TFLite library loads, the model runs,
// and the input conventions are right.
//
// Usage: faceBlendshapesTest <tflite library> <face_blendshapes.tflite> <canonical_face_model.obj>
//
// MediaPipe's canonical_face_model.obj only has 468 vertices, so the 10 iris
// landmarks are synthesized from the eye corners.

#include "faceBlendshapes.h"
#include "tfliteRuntime.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct V3 {
    double x, y, z;
};

int g_failures = 0;

void check(bool ok, const char *what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::vector<V3> readObj(const char *path)
{
    std::vector<V3> v;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2 || line[0] != 'v' || line[1] != ' ') continue;
        std::istringstream ss(line.substr(2));
        V3 p{};
        ss >> p.x >> p.y >> p.z;
        v.push_back(p);
    }
    return v;
}

void addIris(std::vector<V3> &v, int a, int b)
{
    V3 c{(v[a].x + v[b].x) / 2, (v[a].y + v[b].y) / 2, (v[a].z + v[b].z) / 2};
    const double r = 0.2 * std::hypot(v[a].x - v[b].x, v[a].y - v[b].y);
    v.push_back(c);
    v.push_back({c.x + r, c.y, c.z});
    v.push_back({c.x, c.y + r, c.z});
    v.push_back({c.x - r, c.y, c.z});
    v.push_back({c.x, c.y - r, c.z});
}

int indexOf(const char *name)
{
    for (int i = 0; i < fbs::kNumBlendshapes; ++i)
        if (std::string(fbs::kBlendshapeNames[i]) == name) return i;
    return -1;
}

// Y-up mesh -> image orientation, like the node's default inputSpace.
std::array<float, fbs::kNumBlendshapes>
solve(fbs::BlendshapeModel &model, const std::vector<V3> &v, double scale = 1, double offset = 0)
{
    std::array<float, fbs::kNumSubset * 2> xy{};
    for (int i = 0; i < fbs::kNumSubset; ++i) {
        const V3 &p = v[fbs::kLandmarksSubset[i]];
        xy[2 * i] = float(p.x * scale + offset);
        xy[2 * i + 1] = float(-p.y * scale + offset);
    }
    std::array<float, fbs::kNumBlendshapes> s{};
    std::string err;
    if (!model.run(xy.data(), s.data(), err)) std::printf("run failed: %s\n", err.c_str());
    return s;
}

std::vector<V3> squashEye(std::vector<V3> v, const std::vector<int> &ids, int cornerA, int cornerB)
{
    const double cy = (v[cornerA].y + v[cornerB].y) / 2;
    for (int i : ids) v[i].y = cy + (v[i].y - cy) * 0.1;
    return v;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::printf("usage: %s <tflite library> <model.tflite> <canonical_face_model.obj>\n", argv[0]);
        return 2;
    }

    std::string loadedFrom, err;
    if (!tfl::loadLibrary({argv[1]}, loadedFrom, err)) {
        std::printf("could not load library:%s\n", err.c_str());
        return 1;
    }
    std::printf("TFLite %s from %s\n", tfl::api()->TfLiteVersion(), loadedFrom.c_str());

    fbs::BlendshapeModel model;
    if (!model.load(argv[2], err)) {
        std::printf("could not load model: %s\n", err.c_str());
        return 1;
    }

    std::vector<V3> v = readObj(argv[3]);
    if (v.size() == 468) {
        addIris(v, 33, 133);  // subject's right eye -> 468..472
        addIris(v, 362, 263); // subject's left eye  -> 473..477
    }
    check(v.size() == fbs::kNumLandmarks, "mesh has 478 landmarks");
    if (v.size() != fbs::kNumLandmarks) return 1;

    const auto base = solve(model, v);
    bool inRange = true;
    for (float s : base) inRange &= (s >= 0.0f && s <= 1.0f);
    check(inRange, "scores are in [0, 1]");

    const auto moved = solve(model, v, 37.0, 250.0);
    double maxDiff = 0;
    for (int i = 0; i < fbs::kNumBlendshapes; ++i)
        maxDiff = std::fmax(maxDiff, std::fabs(base[i] - moved[i]));
    check(maxDiff < 1e-4, "scale and translation do not change the result");

    const int blinkL = indexOf("eyeBlinkLeft"), blinkR = indexOf("eyeBlinkRight");
    const std::vector<int> leftEye = {263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388,
                                      387, 386, 385, 384, 398, 473, 474, 475, 476, 477};
    const std::vector<int> rightEye = {33,  7,   163, 144, 145, 153, 154, 155, 133, 246, 161,
                                       160, 159, 158, 157, 173, 468, 469, 470, 471, 472};
    const auto closedL = solve(model, squashEye(v, leftEye, 263, 362));
    const auto closedR = solve(model, squashEye(v, rightEye, 33, 133));
    std::printf(
        "  close left eye:  eyeBlinkLeft %.3f eyeBlinkRight %.3f\n", closedL[blinkL],
        closedL[blinkR]
    );
    std::printf(
        "  close right eye: eyeBlinkLeft %.3f eyeBlinkRight %.3f\n", closedR[blinkL],
        closedR[blinkR]
    );
    check(closedL[blinkL] > closedL[blinkR] + 0.2, "closing the subject's left eye drives eyeBlinkLeft");
    check(closedR[blinkR] > closedR[blinkL] + 0.2, "closing the subject's right eye drives eyeBlinkRight");

    std::vector<V3> jaw = v;
    for (V3 &p : jaw)
        if (p.y < v[13].y) p.y -= 1.5;
    const int jawOpen = indexOf("jawOpen");
    const auto opened = solve(model, jaw);
    std::printf("  jawOpen %.3f -> %.3f\n", base[jawOpen], opened[jawOpen]);
    check(opened[jawOpen] > base[jawOpen] + 0.25, "dropping the jaw drives jawOpen");

    int pairs = 0;
    bool mirrorOk = true;
    for (int i = 0; i < fbs::kNumBlendshapes; ++i) {
        const int j = fbs::kMirroredBlendshape[i];
        mirrorOk &= fbs::kMirroredBlendshape[j] == i;
        if (j != i) ++pairs;
    }
    mirrorOk &= fbs::kMirroredBlendshape[blinkL] == blinkR;
    mirrorOk &= fbs::kMirroredBlendshape[indexOf("jawLeft")] == indexOf("jawRight");
    mirrorOk &= fbs::kMirroredBlendshape[jawOpen] == jawOpen;
    std::printf("  %d shapes have a left/right partner\n", pairs);
    check(mirrorOk && pairs == 40, "left/right swap table is consistent");

    model.unload();
    tfl::unloadLibrary();
    std::printf("%s\n", g_failures ? "FAILED" : "ALL PASSED");
    return g_failures ? 1 : 0;
}
