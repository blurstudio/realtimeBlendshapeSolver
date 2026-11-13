#pragma once

#include "faceBlendshapes.h"

#include <maya/MObject.h>
#include <maya/MPlugArray.h>
#include <maya/MPxNode.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>

#include <array>
#include <string>
#include <vector>

// Computes the 52 ARKit-style blendshape weights predicted by MediaPipe's
// face_blendshapes.tflite from a mesh whose vertex order matches the MediaPipe
// face mesh: 478 vertices with iris refinement, or 468 without, in which case the
// irises are synthesized looking straight ahead.
class TfliteArkitSolverNode : public MPxNode {
  public:
    static void *creator();
    static MStatus initialize();

    MStatus compute(const MPlug &plug, MDataBlock &data) override;
    MStatus setDependentsDirty(const MPlug &plug, MPlugArray &affected) override;

    static const MTypeId id;
    static const char *typeName;

    // Folder the plugin was loaded from. The default model is looked for here.
    static MString s_pluginDir;

    // ---- Inputs
    static MObject aInMesh;
    static MObject aInIrisMesh;
    static MObject aModelPath;
    static MObject aInputSpace;
    static MObject aInputMatrix;
    static MObject aLandmarkScale;
    static MObject aSwapLeftRight;

    // ---- Outputs
    static MObject aValid;
    static std::array<MObject, fbs::kNumBlendshapes> aBlendshapes;

  private:
    bool ensureModel(const std::string &path);
    void reportOnce(const std::string &msg);

    fbs::BlendshapeModel m_model;
    std::string m_failedPath; // don't retry a bad path every frame
    std::string m_lastReported;

    static std::vector<MObject> s_outputs;
};
