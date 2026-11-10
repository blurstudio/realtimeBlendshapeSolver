#include "tfliteArkitSolverNode.h"
#include "version.h"

#include <maya/MFnPlugin.h>

MStatus initializePlugin(MObject obj)
{
    MFnPlugin plugin(obj, "Blur Studio", VERSION_STRING, "Any");
    TfliteArkitSolverNode::s_pluginDir = plugin.loadPath();

    MStatus status = plugin.registerNode(
        TfliteArkitSolverNode::typeName, TfliteArkitSolverNode::id,
        TfliteArkitSolverNode::creator, TfliteArkitSolverNode::initialize
    );
    CHECK_MSTATUS_AND_RETURN_IT(status);
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj)
{
    MFnPlugin plugin(obj);
    MStatus status = plugin.deregisterNode(TfliteArkitSolverNode::id);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    return MS::kSuccess;
}
