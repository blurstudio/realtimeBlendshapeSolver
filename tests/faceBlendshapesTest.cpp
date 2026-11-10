// Standalone (no Maya) check that the model runs, matches TensorFlow Lite, and
// the input conventions are right.
//
// Usage: faceBlendshapesTest <face_blendshapes.tflite> <canonical_face_model.obj>
//
// MediaPipe's canonical_face_model.obj only has 468 vertices, so this also
// covers the no-iris path: fbs::synthesizeIris fills in the 10 iris landmarks.

#include "faceBlendshapes.h"

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

// TensorFlow Lite 2.22 (LiteRt 2.3.0) scores for the canonical face with
// synthesized irises, i.e. solve(model, mesh) below.
// clang-format off
const float kTfLiteCanonicalScores[fbs::kNumBlendshapes] = {
    0.0000052f, 0.3392323f, 0.2780324f, 0.0160076f,
    0.0033426f, 0.0025734f, 0.0000673f, 0.0000002f,
    0.0000003f, 0.4309839f, 0.3951717f, 0.1873899f,
    0.2031238f, 0.0322983f, 0.1399942f, 0.1395191f,
    0.0328343f, 0.0483988f, 0.0530597f, 0.2427855f,
    0.1895487f, 0.0013256f, 0.0006257f, 0.0000617f,
    0.0019357f, 0.0757132f, 0.0001366f, 0.0166299f,
    0.0328260f, 0.0292926f, 0.0000162f, 0.0000218f,
    0.0033500f, 0.0026560f, 0.0014063f, 0.0014451f,
    0.1493420f, 0.1478278f, 0.0096107f, 0.0013651f,
    0.1527375f, 0.0730304f, 0.0415678f, 0.0492742f,
    0.0013531f, 0.0016791f, 0.0001054f, 0.0004019f,
    0.0032066f, 0.0040961f, 0.0000004f, 0.0000005f,
};
// clang-format on

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
        if (fbs::kLandmarksSubset[i] >= int(v.size())) continue; // iris, synthesized below
        const V3 &p = v[fbs::kLandmarksSubset[i]];
        xy[2 * i] = float(p.x * scale + offset);
        xy[2 * i + 1] = float(-p.y * scale + offset);
    }
    if (v.size() == fbs::kNumFaceLandmarks) fbs::synthesizeIris(xy.data());
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
    if (argc < 3) {
        std::printf("usage: %s <model.tflite> <canonical_face_model.obj>\n", argv[0]);
        return 2;
    }

    std::string err;
    fbs::BlendshapeModel model;
    if (!model.load(argv[1], err)) {
        std::printf("could not load model: %s\n", err.c_str());
        return 1;
    }

    fbs::BlendshapeModel notAModel;
    check(!notAModel.load(argv[2], err) && !notAModel.isLoaded(), "a non-model file fails to load");
    std::printf("  %s\n", err.c_str());

    const std::vector<V3> v = readObj(argv[2]);
    check(v.size() == fbs::kNumFaceLandmarks, "mesh has 468 landmarks (no iris)");
    if (v.size() != fbs::kNumFaceLandmarks) return 1;

    const auto base = solve(model, v);
    bool inRange = true;
    for (float s : base) inRange &= (s >= 0.0f && s <= 1.0f);
    check(inRange, "scores are in [0, 1]");

    double tfliteDiff = 0;
    for (int i = 0; i < fbs::kNumBlendshapes; ++i)
        tfliteDiff = std::fmax(tfliteDiff, std::fabs(base[i] - kTfLiteCanonicalScores[i]));
    std::printf("  largest difference from TensorFlow Lite: %.2g\n", tfliteDiff);
    check(tfliteDiff < 1e-4, "scores match TensorFlow Lite");

    const auto moved = solve(model, v, 37.0, 250.0);
    double maxDiff = 0;
    for (int i = 0; i < fbs::kNumBlendshapes; ++i)
        maxDiff = std::fmax(maxDiff, std::fabs(base[i] - moved[i]));
    check(maxDiff < 1e-4, "scale and translation do not change the result");

    double maxLook = 0;
    for (int i = 0; i < fbs::kNumBlendshapes; ++i)
        if (std::string(fbs::kBlendshapeNames[i]).rfind("eyeLook", 0) == 0)
            maxLook = std::fmax(maxLook, base[i]);
    std::printf("  largest eyeLook* with synthesized irises: %.3f\n", maxLook);
    check(maxLook < 0.3, "synthesized irises look straight ahead");

    const int blinkL = indexOf("eyeBlinkLeft"), blinkR = indexOf("eyeBlinkRight");
    const std::vector<int> leftEye = {263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388,
                                      387, 386, 385, 384, 398};
    const std::vector<int> rightEye = {33,  7,   163, 144, 145, 153, 154, 155, 133, 246, 161,
                                       160, 159, 158, 157, 173};
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
    std::printf("%s\n", g_failures ? "FAILED" : "ALL PASSED");
    return g_failures ? 1 : 0;
}
