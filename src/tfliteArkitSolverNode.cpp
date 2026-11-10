#include "tfliteArkitSolverNode.h"

#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnMesh.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MMatrix.h>
#include <maya/MPlug.h>
#include <maya/MPoint.h>

#include <algorithm>
#include <cstdio>
#include <utility>

// NOTE: 0x00000 - 0x7ffff is Autodesk's range for local/internal plugins.
// Replace with an ID from your studio's registered block before distribution.
const MTypeId TfliteArkitSolverNode::id(0x0007F0A2);
const char *TfliteArkitSolverNode::typeName = "tfliteArkitSolver";

MString TfliteArkitSolverNode::s_pluginDir;

MObject TfliteArkitSolverNode::aInMesh;
MObject TfliteArkitSolverNode::aModelPath;
MObject TfliteArkitSolverNode::aInputSpace;
MObject TfliteArkitSolverNode::aInputMatrix;
MObject TfliteArkitSolverNode::aLandmarkScale;
MObject TfliteArkitSolverNode::aSwapLeftRight;

MObject TfliteArkitSolverNode::aValid;
std::array<MObject, fbs::kNumBlendshapes> TfliteArkitSolverNode::aBlendshapes;

std::vector<MObject> TfliteArkitSolverNode::s_outputs;

namespace {

const char *kDefaultModelName = "face_blendshapes.tflite";

enum InputSpace : short {
    // +Y up, +Z toward the camera. Maya convention, and the orientation of
    // MediaPipe's canonical_face_model.obj
    kInputSpaceYUp = 0,
    // MediaPipe image orientation: +Y down, +Z away from the camera.
    // Use this when the cache holds raw MediaPipe landmark values.
    kInputSpaceImage = 1,
};

} // namespace

void *TfliteArkitSolverNode::creator() { return new TfliteArkitSolverNode(); }

MStatus TfliteArkitSolverNode::initialize()
{
    MFnTypedAttribute tAttr;
    MFnNumericAttribute nAttr;
    MFnEnumAttribute eAttr;
    MFnMatrixAttribute mAttr;

    // ------------------------------------------------------------------ Inputs

    aInMesh = tAttr.create("inMesh", "im", MFnData::kMesh);
    tAttr.setStorable(false);

    // Empty means "face_blendshapes.tflite next to the plugin".
    aModelPath = tAttr.create("modelPath", "mp", MFnData::kString);
    tAttr.setUsedAsFilename(true);

    aInputSpace = eAttr.create("inputSpace", "isp", kInputSpaceYUp);
    eAttr.addField("yUp", kInputSpaceYUp);
    eAttr.addField("mediaPipeImage", kInputSpaceImage);
    eAttr.setKeyable(true);

    // Applied to the points before they are projected onto XY. Connect a
    // camera's worldInverseMatrix (with the mesh's worldMesh) to solve from
    // that camera's point of view. The model only sees x/y, so head rotation
    // relative to the projection axis matters.
    aInputMatrix = mAttr.create("inputMatrix", "imat");

    // Multiplies x/y after the axis conversion. Only the aspect ratio matters,
    // because the model normalizes position and scale itself. For raw
    // MediaPipe normalized landmarks, set this to (imageWidth, imageHeight).
    {
        MObject x = nAttr.create("landmarkScaleX", "lsx", MFnNumericData::kDouble, 1.0);
        MObject y = nAttr.create("landmarkScaleY", "lsy", MFnNumericData::kDouble, 1.0);
        aLandmarkScale = nAttr.create("landmarkScale", "ls", x, y);
        nAttr.setDefault(1.0, 1.0);
        nAttr.setKeyable(true);
    }

    // Swap every *Left / *Right output pair, e.g. when the landmarks came from
    // mirrored "selfie" footage. Negating X would not do this: the model
    // identifies landmarks by index, not by which side of the image they're on.
    aSwapLeftRight = nAttr.create("swapLeftRight", "slr", MFnNumericData::kBoolean, false);
    nAttr.setKeyable(true);

    // ----------------------------------------------------------------- Outputs

    aValid = nAttr.create("valid", "vld", MFnNumericData::kBoolean, false);
    nAttr.setWritable(false);
    nAttr.setStorable(false);

    s_outputs = {aValid};
    for (int i = 0; i < fbs::kNumBlendshapes; ++i) {
        MString longName(fbs::kBlendshapeNames[i]);
        if (i == 0) longName = "neutral"; // "_neutral" is not a friendly plug name
        char shortName[8];
        std::snprintf(shortName, sizeof(shortName), "bs%02d", i);
        aBlendshapes[i] = nAttr.create(longName, shortName, MFnNumericData::kDouble, 0.0);
        nAttr.setWritable(false);
        nAttr.setStorable(false);
        s_outputs.push_back(aBlendshapes[i]);
    }

    const MObject inputs[] = {
        aInMesh, aModelPath, aInputSpace, aInputMatrix, aLandmarkScale, aSwapLeftRight,
    };

    MStatus status;
    for (const MObject &attr : inputs) {
        status = addAttribute(attr);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    for (const MObject &attr : s_outputs) {
        status = addAttribute(attr);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    for (const MObject &in : inputs) {
        for (const MObject &out : s_outputs) {
            status = attributeAffects(in, out);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
    }
    return MS::kSuccess;
}

void TfliteArkitSolverNode::reportOnce(const std::string &msg)
{
    if (msg == m_lastReported) return;
    m_lastReported = msg;
    MGlobal::displayError(MString(typeName) + ": " + msg.c_str());
}

bool TfliteArkitSolverNode::ensureModel(const std::string &path)
{
    if (m_model.isLoaded() && m_model.path() == path) return true;
    if (path == m_failedPath) return false;

    std::string err;
    if (m_model.load(path, err)) {
        m_failedPath.clear();
        m_lastReported.clear();
        return true;
    }
    m_failedPath = path;
    reportOnce(err);
    return false;
}

MStatus TfliteArkitSolverNode::setDependentsDirty(const MPlug &plug, MPlugArray &affected)
{
    // Setting modelPath again (even to the same value) retries a failed load.
    if (plug == aModelPath) {
        m_failedPath.clear();
        m_lastReported.clear();
    }
    return MPxNode::setDependentsDirty(plug, affected);
}

MStatus TfliteArkitSolverNode::compute(const MPlug &plug, MDataBlock &data)
{
    const MObject attr = plug.isChild() ? plug.parent().attribute() : plug.attribute();
    if (std::find(s_outputs.begin(), s_outputs.end(), attr) == s_outputs.end()) {
        return MS::kUnknownParameter;
    }

    // ---- Gather inputs
    std::string modelPath = data.inputValue(aModelPath).asString().asChar();
    if (modelPath.empty()) {
        modelPath = (s_pluginDir + "/" + kDefaultModelName).asChar();
    }
    const short inputSpace = data.inputValue(aInputSpace).asShort();
    const MMatrix mat = data.inputValue(aInputMatrix).asMatrix();
    const double2 &scale = data.inputValue(aLandmarkScale).asDouble2();
    const bool swapLeftRight = data.inputValue(aSwapLeftRight).asBool();

    // Axis conversion into MediaPipe image orientation (+x right, +y down).
    // YUp -> image flips Y only. With the projection along Z this is a view of
    // the face from the front, so it does not mirror left/right.
    const double sx = scale[0];
    const double sy = scale[1] * ((inputSpace == kInputSpaceYUp) ? -1.0 : 1.0);
    const bool identity = mat.isEquivalent(MMatrix::identity);

    std::array<float, fbs::kNumSubset * 2> xy{};
    std::array<float, fbs::kNumBlendshapes> scores{};
    bool valid = false;

    MObject meshObj = data.inputValue(aInMesh).asMesh();
    if (!meshObj.isNull()) {
        MStatus status;
        MFnMesh fnMesh(meshObj, &status);
        const int numVerts = status ? fnMesh.numVertices() : 0;
        const float *pts = status ? fnMesh.getRawPoints(&status) : nullptr;

        if (!status || pts == nullptr || numVerts == 0) {
            // Nothing connected yet; no need to complain.
        }
        else if (numVerts != fbs::kNumLandmarks && numVerts != fbs::kNumFaceLandmarks) {
            reportOnce(
                "inMesh has " + std::to_string(numVerts) + " vertices; expected " +
                std::to_string(fbs::kNumFaceLandmarks) + " (MediaPipe face mesh) or " +
                std::to_string(fbs::kNumLandmarks) + " (with iris refinement)"
            );
        }
        else if (ensureModel(modelPath)) {
            for (int i = 0; i < fbs::kNumSubset; ++i) {
                if (fbs::kLandmarksSubset[i] >= numVerts) continue; // iris, synthesized below
                const float *p = pts + 3 * fbs::kLandmarksSubset[i];
                double x = p[0], y = p[1];
                if (!identity) {
                    MPoint q = MPoint(p[0], p[1], p[2]) * mat;
                    if (q.w != 0.0 && q.w != 1.0) q.cartesianize();
                    x = q.x;
                    y = q.y;
                }
                xy[2 * i] = static_cast<float>(x * sx);
                xy[2 * i + 1] = static_cast<float>(y * sy);
            }
            if (numVerts == fbs::kNumFaceLandmarks) fbs::synthesizeIris(xy.data());
            std::string err;
            valid = m_model.run(xy.data(), scores.data(), err);
            if (!valid) reportOnce(err);
        }
    }
    if (!valid) scores.fill(0.0f);
    if (valid && swapLeftRight) {
        for (int i = 0; i < fbs::kNumBlendshapes; ++i) {
            const int j = fbs::kMirroredBlendshape[i];
            if (j > i) std::swap(scores[i], scores[j]);
        }
    }

    // ---- Write outputs
    {
        MDataHandle h = data.outputValue(aValid);
        h.setBool(valid);
        h.setClean();
    }
    for (int i = 0; i < fbs::kNumBlendshapes; ++i) {
        MDataHandle h = data.outputValue(aBlendshapes[i]);
        h.setDouble(scores[i]);
        h.setClean();
    }

    data.setClean(plug);
    return MS::kSuccess;
}
