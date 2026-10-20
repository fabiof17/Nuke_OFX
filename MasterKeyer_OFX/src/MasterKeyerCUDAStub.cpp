// MasterKeyerCUDAStub.cpp
// Used when the plug-in is built without the CUDA toolkit: every render runs on the CPU.
#include "MasterKeyerCore.h"

namespace mk {

bool cudaIsCompiledIn() { return false; }
bool cudaDeviceInfo(std::string& description)
{
    description = "CUDA: not compiled in this build (CPU rendering)";
    return false;
}
bool processCUDA(const Params&, const Inputs&, float*, std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}
bool processCUDARows(const Params&, const RowInputs&, const RowWriter&, std::string& error)
{
    error = "CUDA not compiled in this build";
    return false;
}
void cudaReleaseResources() {}
void cudaSetProfiling(bool) {}
std::string cudaLastProfile() { return std::string(); }

} // namespace mk
