// DespillITCore.h
// Back-end entry points of despill_IT_OFX. All images cover the same window (p.width x
// p.height), dense, row-major, bottom-up as in OFX:
//   srcRGBA  : plate, 4 floats per pixel              key     : 1 float (alpha of a keyer) or null
//   cleanRGB : clean plate, 3 floats, or null          bgRGB   : background, 3 floats, or null
//   protect  : 1 float (1 = protected), or null        mask    : 1 float (already inverted), or null
//   outRGBA  : result, 4 floats per pixel
#pragma once

#include <functional>
#include <string>

#include "DespillITCommon.h"

namespace dspit {

struct Inputs {
    const float* src = nullptr;
    const float* key = nullptr;
    const float* clean = nullptr;
    const float* bg = nullptr;
    const float* protect = nullptr;
    const float* mask = nullptr;
};

void processCPU(const Params& p, const Inputs& in, float* outRGBA);

// CUDA back-end (DespillITCUDA.cu, or stubs when built without CUDA).
bool cudaIsCompiledIn();
bool cudaDeviceInfo(std::string& description);
bool processCUDA(const Params& p, const Inputs& in, float* outRGBA, std::string& error);
// Same, streaming rows straight from / into the transfer memory. Each reader writes rows
// [y0, y1) of its image densely (channel counts as above); null readers = absent inputs.
// Called from several threads at once, on disjoint rows.
using RowReader = std::function<void(int y0, int y1, float* rows)>;
using RowWriter = std::function<void(int y0, int y1, const float* rows)>;
struct RowInputs {
    const RowReader* src = nullptr;
    const RowReader* key = nullptr;
    const RowReader* clean = nullptr;
    const RowReader* bg = nullptr;
    const RowReader* protect = nullptr;
    const RowReader* mask = nullptr;
};
bool processCUDARows(const Params& p, const RowInputs& in, const RowWriter& outRGBA, std::string& error);
void cudaReleaseResources();
void cudaSetProfiling(bool on);
std::string cudaLastProfile();

// What the pipeline needs, from the parameters and the connected inputs.
inline bool needsLocalScreen(const Params& p) { return p.screenSource == kScreenFromPlate; }
inline bool needsCore(const Params& p) { return p.edgeRestore > 0.f || p.output == kOutCore; }
inline bool needsBGBlur(const Params& p) { return p.hasBG && (p.spillLight == kLightBackground || p.wrapAmount > 0.f); }
inline bool needsWrap(const Params& p) { return p.hasBG && p.wrapAmount > 0.f; }

} // namespace dspit
