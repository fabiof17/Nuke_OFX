// DespillCore.h
// Back-end entry points of L_Despill_OFX. Images are dense, row-major, bottom-up as in OFX.
//   srcRGBA : 4 floats per pixel over the whole buffer (p.bufW x p.bufH, output window plus
//             the margin the spatial filters need, edges repeated)
//   bgRGB   : 3 floats per pixel over the output window, or nullptr
//   mask    : 1 float per pixel over the output window (already inverted), or nullptr
//   outRGBA : 4 floats per pixel over the output window (p.winW x p.winH)
#pragma once

#include <functional>
#include <string>

#include "DespillCommon.h"

namespace ldespill {

void processCPU(const Params& p, const float* srcRGBA, const float* bgRGB, const float* mask, float* outRGBA);

// CUDA back-end (DespillCUDA.cu, or stubs when built without CUDA).
bool cudaIsCompiledIn();
bool cudaDeviceInfo(std::string& description);
bool processCUDA(const Params& p, const float* srcRGBA, const float* bgRGB, const float* mask, float* outRGBA,
                 std::string& error);
// Same, streaming rows straight from / into the transfer memory: the readers write rows
// [y0, y1) of their image densely, the writer receives the result rows of the window.
// They are called from several threads at once, on disjoint rows.
using RowReader = std::function<void(int y0, int y1, float* rows)>;
using RowWriter = std::function<void(int y0, int y1, const float* rows)>;
bool processCUDARows(const Params& p, const RowReader& srcRGBA, const RowReader* bgRGB, const RowReader* mask,
                     const RowWriter& outRGBA, std::string& error);
void cudaReleaseResources();
// Per-stage GPU timings (benchmark tool only).
void cudaSetProfiling(bool on);
std::string cudaLastProfile();

} // namespace ldespill
