#pragma once

// Maya-independent wrapper around MediaPipe's face_blendshapes.tflite model.
//
// Model contract (matches mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc):
//   input  float32 [1, 146, 2]  x,y of kLandmarksSubset, in image orientation
//                               (+x right, +y down). MediaPipe passes pixel coordinates,
//                               but the model normalizes internally (subtract mean,
//                               divide by norm), so translation and uniform scale have
//                               no effect. Aspect ratio and mirroring DO matter.
//   output float32 [52]         blendshape scores in [0,1], order kBlendshapeNames.
//
// *Left / *Right refer to the subject's left and right, not the viewer's.

#include <array>
#include <string>

namespace tfl {
struct TfLiteModel;
struct TfLiteInterpreter;
} // namespace tfl

namespace fbs {

constexpr int kNumLandmarks = 478;     // 468 face mesh + 10 iris
constexpr int kNumFaceLandmarks = 468; // face mesh without iris refinement
constexpr int kNumSubset = 146;
constexpr int kNumBlendshapes = 52;

extern const std::array<int, kNumSubset> kLandmarksSubset;

// For meshes without iris refinement: fills the iris entries (landmarks 468-477)
// of a kLandmarksSubset xy array, already in image space, from the eye corners.
// The irises look straight ahead, so gaze is not tracked: eyeLook* stay near rest.
void synthesizeIris(float *xy);
extern const std::array<const char *, kNumBlendshapes> kBlendshapeNames;
// kMirroredBlendshape[i] is the index of the *Right shape for a *Left shape and
// vice versa, or i itself for unpaired shapes (jawOpen, browInnerUp, ...).
extern const std::array<int, kNumBlendshapes> kMirroredBlendshape;

class BlendshapeModel {
  public:
    BlendshapeModel() = default;
    ~BlendshapeModel();
    BlendshapeModel(const BlendshapeModel &) = delete;
    BlendshapeModel &operator=(const BlendshapeModel &) = delete;

    // Requires tfl::loadLibrary to have succeeded. Validates tensor shapes.
    bool load(const std::string &modelPath, std::string &error);
    void unload();
    bool isLoaded() const { return m_interpreter != nullptr; }
    const std::string &path() const { return m_path; }

    // xy: kNumSubset * 2 floats. scores: kNumBlendshapes floats.
    bool run(const float *xy, float *scores, std::string &error);

  private:
    tfl::TfLiteModel *m_model = nullptr;
    tfl::TfLiteInterpreter *m_interpreter = nullptr;
    std::string m_path;
    std::string m_lastTfError;
};

} // namespace fbs
