/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX. GPL v2 or later, see LICENSE.
 * ***** END LICENSE BLOCK ***** */

// Used when the plugin is built without CUDA (CORNERPINEX_OFX_WITH_CUDA=OFF):
// everything is rendered on the CPU.

#include <cstdio>
#include <cstdlib>

#include "cpEngine.h"

namespace cpg {

bool cudaCompiledIn()
{
    return false;
}

CudaDeviceInfo cudaQueryDevice(int /*requestedDevice*/)
{
    CudaDeviceInfo info;
    info.available = false;
    info.device = -1;
    info.totalMem = 0;
    info.message = "Built without CUDA support";
    return info;
}

bool cudaRender(const RenderJob& /*job*/, int /*device*/, std::string* err)
{
    if (err) {
        *err = "Built without CUDA support";
    }
    return false;
}
bool profilingEnabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("CORNERPINEX_OFX_PROFILE");
        enabled = (v && *v && *v != '0') ? 1 : 0;
    }
    return enabled == 1;
}

void profileCpuRender(const char* what, int w, int h, double ms)
{
    if ( !profilingEnabled() ) {
        return;
    }
    const char* tmp = std::getenv("TEMP");
    if (!tmp) {
        tmp = std::getenv("TMP");
    }
    std::string path = std::string(tmp ? tmp : "/tmp") + "/CornerPinEx_OFX_profile.log";
    FILE* f = std::fopen(path.c_str(), "a");
    if (f) {
        std::fprintf(f, "CPU render (%s) %dx%d total=%gms\n", what, w, h, ms);
        std::fclose(f);
    }
}
} // namespace cpg
