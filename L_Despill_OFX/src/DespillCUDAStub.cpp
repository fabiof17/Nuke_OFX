// DespillCUDAStub.cpp
// Used when the plug-in is built without the CUDA toolkit: the GPU option stays visible
// but every render falls back to the CPU.
#include "DespillCore.h"

namespace ldespill {

bool cudaIsCompiledIn() { return false; }

bool cudaDeviceInfo(std::string& description)
{
    description = "CUDA: not compiled in this build (CPU rendering)";
    return false;
}

bool processCUDA(const Params&, const float*, const float*, const float*, float*, std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}

bool processCUDARows(const Params&, const RowReader&, const RowReader*, const RowReader*, const RowWriter&,
                     std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}

void cudaReleaseResources() {}
void cudaSetProfiling(bool) {}
std::string cudaLastProfile() { return std::string(); }

} // namespace ldespill
