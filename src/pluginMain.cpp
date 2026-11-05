#include "tfliteArkitSolverNode.h"
#include "tfliteRuntime.h"
#include "version.h"

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Search order:
//   1. $MAYA_TFLITE_LIBRARY (full path to the library file)
//   2. Next to the plugin: libLiteRt.dll, tensorflowlite_c.dll, ...
//   3. The same names on the normal system library search path
std::vector<std::string> libraryCandidates(const MString &pluginDir)
{
    std::vector<std::string> out;
    if (const char *env = std::getenv("MAYA_TFLITE_LIBRARY")) {
        if (*env) out.emplace_back(env);
    }
    const std::vector<std::string> names = tfl::defaultLibraryNames();
    for (const std::string &n : names) out.push_back(std::string(pluginDir.asChar()) + "/" + n);
    for (const std::string &n : names) out.push_back(n);
    return out;
}

} // namespace

MStatus initializePlugin(MObject obj)
{
    MFnPlugin plugin(obj, "Blur Studio", VERSION_STRING, "Any");
    TfliteArkitSolverNode::s_pluginDir = plugin.loadPath();

    std::string loadedFrom, error;
    if (!tfl::loadLibrary(
            libraryCandidates(TfliteArkitSolverNode::s_pluginDir), loadedFrom, error
        )) {
        MGlobal::displayError(
            MString("tfliteArkitSolver: could not load the TensorFlow Lite C library. "
                    "Put libLiteRt.dll next to the plugin or set MAYA_TFLITE_LIBRARY. Tried:") +
            error.c_str()
        );
        return MS::kFailure;
    }
    MGlobal::displayInfo(
        MString("tfliteArkitSolver: using TensorFlow Lite ") + tfl::api()->TfLiteVersion() +
        " from " + loadedFrom.c_str()
    );

    MStatus status = plugin.registerNode(
        TfliteArkitSolverNode::typeName, TfliteArkitSolverNode::id,
        TfliteArkitSolverNode::creator, TfliteArkitSolverNode::initialize
    );
    if (!status) tfl::unloadLibrary();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj)
{
    MFnPlugin plugin(obj);
    MStatus status = plugin.deregisterNode(TfliteArkitSolverNode::id);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    // All node instances (and their interpreters) are gone once deregistration succeeds.
    tfl::unloadLibrary();
    return MS::kSuccess;
}
