// push_IT_OFXCore.h
// Back-end entry points. Buffers are dense, row-major, bottom-up as in OFX,
// width * height pixels:
//   srcRGBA : 4 floats per pixel (as delivered by the host)
//   matte   : 1 float per pixel  (already resolved from the chosen channel)
//   vecRGBA : 4 floats per pixel, or nullptr when there is no vector input
//   dstRGBA : 4 floats per pixel
#pragma once

#include <functional>
#include <string>
#include "push_IT_OFXCommon.h"

namespace pushit {

void processCPU(const Params& p, const float* srcRGBA, const float* matte,
                const float* vecRGBA, float* dstRGBA);

// CUDA back-end (push_IT_OFXCUDA.cu, or stubs when built without CUDA).
bool cudaIsCompiledIn();
bool cudaDeviceInfo(std::string& description);          // false if no usable device
bool processCUDA(const Params& p, const float* srcRGBA, const float* matte,
                 const float* vecRGBA, float* dstRGBA, std::string& error);
// Same, streaming rows: the readers write rows [y0, y1) (width pixels each, densely packed)
// and the writer receives the result rows, straight from / into the transfer memory, so the
// plug-in converts Nuke's images there without an intermediate copy. Readers and writer are
// called from several threads at once (on disjoint rows). matte == nullptr: source alpha.
using RowReader = std::function<void(int y0, int y1, float* rows)>;
using RowWriter = std::function<void(int y0, int y1, const float* rows)>;
bool processCUDARows(const Params& p, const RowReader& srcRGBA, const RowReader* matte,
                     const RowReader* vecRGBA, const RowWriter& dstRGBA, std::string& error);
void cudaReleaseResources();
// Per-stage GPU timings (benchmark tool only).
void cudaSetProfiling(bool on);
std::string cudaLastProfile();

} // namespace pushit
