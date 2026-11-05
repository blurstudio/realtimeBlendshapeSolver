#include "tfliteRuntime.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tfl {

namespace {

void *g_handle = nullptr;
Api g_api{};
bool g_loaded = false;

#ifdef _WIN32
std::wstring widen(const std::string &s)
{
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

bool isPath(const std::string &s)
{
    return s.find('/') != std::string::npos || s.find('\\') != std::string::npos;
}

void *openLibrary(const std::string &name, std::string &why)
{
    // For a full path, let the library's own dependencies resolve from its folder.
    const DWORD flags = isPath(name) ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;
    HMODULE h = LoadLibraryExW(widen(name).c_str(), nullptr, flags);
    if (!h) why = "LoadLibrary error " + std::to_string(GetLastError());
    return (void *)h;
}

void *findSymbol(void *handle, const char *name)
{
    return (void *)GetProcAddress((HMODULE)handle, name);
}

void closeLibrary(void *handle) { FreeLibrary((HMODULE)handle); }
#else
void *openLibrary(const std::string &name, std::string &why)
{
    void *h = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char *e = dlerror();
        why = e ? e : "dlopen failed";
    }
    return h;
}

void *findSymbol(void *handle, const char *name) { return dlsym(handle, name); }

void closeLibrary(void *handle) { dlclose(handle); }
#endif

template <typename Fn> bool bind(void *handle, Fn &slot, const char *name, std::string &missing)
{
    slot = reinterpret_cast<Fn>(findSymbol(handle, name));
    if (!slot) {
        if (!missing.empty()) missing += ", ";
        missing += name;
    }
    return slot != nullptr;
}

bool bindAll(void *h, Api &a, std::string &missing)
{
    bool ok = true;
#define TFL_BIND(fn) ok &= bind(h, a.fn, #fn, missing)
    TFL_BIND(TfLiteVersion);
    TFL_BIND(TfLiteModelCreateFromFile);
    TFL_BIND(TfLiteModelDelete);
    TFL_BIND(TfLiteInterpreterOptionsCreate);
    TFL_BIND(TfLiteInterpreterOptionsDelete);
    TFL_BIND(TfLiteInterpreterOptionsSetNumThreads);
    TFL_BIND(TfLiteInterpreterOptionsSetErrorReporter);
    TFL_BIND(TfLiteInterpreterCreate);
    TFL_BIND(TfLiteInterpreterDelete);
    TFL_BIND(TfLiteInterpreterAllocateTensors);
    TFL_BIND(TfLiteInterpreterInvoke);
    TFL_BIND(TfLiteInterpreterGetInputTensorCount);
    TFL_BIND(TfLiteInterpreterGetOutputTensorCount);
    TFL_BIND(TfLiteInterpreterGetInputTensor);
    TFL_BIND(TfLiteInterpreterGetOutputTensor);
    TFL_BIND(TfLiteTensorType);
    TFL_BIND(TfLiteTensorNumDims);
    TFL_BIND(TfLiteTensorDim);
    TFL_BIND(TfLiteTensorByteSize);
    TFL_BIND(TfLiteTensorCopyFromBuffer);
    TFL_BIND(TfLiteTensorCopyToBuffer);
#undef TFL_BIND
    return ok;
}

} // namespace

std::vector<std::string> defaultLibraryNames()
{
#if defined(_WIN32)
    return {"libLiteRt.dll", "tensorflowlite_c.dll", "libtensorflowlite_c.dll"};
#elif defined(__APPLE__)
    return {"libLiteRt.dylib", "libtensorflowlite_c.dylib"};
#else
    return {"libLiteRt.so", "libtensorflowlite_c.so"};
#endif
}

bool loadLibrary(
    const std::vector<std::string> &candidates, std::string &loadedFrom, std::string &error
)
{
    if (g_loaded) return true;
    error.clear();
    for (const std::string &name : candidates) {
        if (name.empty()) continue;
        std::string why;
        void *h = openLibrary(name, why);
        if (!h) {
            error += "\n  " + name + ": " + why;
            continue;
        }
        Api a{};
        std::string missing;
        if (!bindAll(h, a, missing)) {
            error += "\n  " + name + ": missing symbols " + missing;
            closeLibrary(h);
            continue;
        }
        g_handle = h;
        g_api = a;
        g_loaded = true;
        loadedFrom = name;
        return true;
    }
    return false;
}

void unloadLibrary()
{
    if (g_handle) closeLibrary(g_handle);
    g_handle = nullptr;
    g_api = Api{};
    g_loaded = false;
}

const Api *api() { return g_loaded ? &g_api : nullptr; }

} // namespace tfl
