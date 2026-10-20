// GlowCUDAStub.cpp
// Used when the plug-in is built without the CUDA toolkit: the GPU option
// stays visible but every render falls back to the CPU.
#include "GlowCore.h"

namespace expoglow {

bool cudaIsCompiledIn() { return false; }

bool cudaDeviceInfo(std::string& description)
{
    description = "CUDA: not compiled in this build (CPU rendering)";
    return false;
}

bool computeGlowCUDA(const PrepParams&, const std::vector<StepPlan>&, const float*, const float*, float*, std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}

bool computeGlowCUDARows(const PrepParams&, const std::vector<StepPlan>&, const RowReader&, const RowReader*,
                         int, int, int, int, const RowWriter&, std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}

void cudaReleaseResources() {}
void cudaSetProfiling(bool) {}
std::string cudaLastProfile() { return std::string(); }

} // namespace expoglow
