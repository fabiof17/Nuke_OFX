// MasterKeyerCore.h
// Back-end entry points of MasterKeyer_OFX. All images cover the same window (p.width x
// p.height), dense, row-major, bottom-up as in OFX:
//   src     : plate, 4 floats per pixel          bg      : background, 3 floats, or null
//   keyIn   : plate used for the matte (Key In), 3 floats, or null (the matte uses src)
//   garbage : 1 float (1 = transparent), or null  holdout : 1 float (1 = opaque), or null
//   out     : result, 4 floats per pixel
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <string>

#include "MasterKeyerCommon.h"

namespace mk {

struct Inputs {
    const float* src = nullptr;
    const float* bg = nullptr;
    const float* keyIn = nullptr;
    const float* garbage = nullptr;
    const float* holdout = nullptr;
};

void processCPU(const Params& p, const Inputs& in, float* out);

bool cudaIsCompiledIn();
bool cudaDeviceInfo(std::string& description);
bool processCUDA(const Params& p, const Inputs& in, float* out, std::string& error);
using RowReader = std::function<void(int y0, int y1, float* rows)>;
using RowWriter = std::function<void(int y0, int y1, const float* rows)>;
struct RowInputs {
    const RowReader* src = nullptr;
    const RowReader* bg = nullptr;
    const RowReader* keyIn = nullptr;        // 3 floats per pixel
    const RowReader* garbage = nullptr;
    const RowReader* holdout = nullptr;
};
// Rows are streamed straight from / into the transfer memory; readers and writer are called
// from several threads at once, on disjoint rows. The GPU only needs the RGB of the plate:
// the src reader gives 3 floats per pixel (the alpha of the plate is not used).
bool processCUDARows(const Params& p, const RowInputs& in, const RowWriter& out, std::string& error);
void cudaReleaseResources();
void cudaSetProfiling(bool on);
std::string cudaLastProfile();

// What the pipeline needs.
inline bool needsDistance(const Params& p) { return (p.autoCC && p.spillStrength > 0.f) || p.edgeSize > 0.f || p.output == kOutEdge; }
inline bool needsBGBlur(const Params& p) { return p.hasBG && p.edgeSize > 0.f && p.edgeFromBG > 0.f; }

// Grain size (degrain Size) measured on a flat, grainy area (rgb: 3 floats per pixel, w x h):
// lag-1 autocorrelation of the chroma (Cb, Cr) once a plane is removed from each channel. Grain
// made of white noise blurred by a gaussian of sigma g has an autocorrelation exp(-d^2 / 4g^2);
// the degrain blur needs a sigma of about 2g, i.e. a Size of about 4g (Size = 2 x sigma).
// Returns 0 when the area is too small (under 6 x 6 pixels) or flat.
inline double estimateGrainSize(const float* rgb, int w, int h)
{
    if (w < 6 || h < 6) return 0.0;
    const size_t N = (size_t)w * h;
    std::vector<double> ch[2] = { std::vector<double>(N), std::vector<double>(N) };
    for (size_t i = 0; i < N; ++i) {
        float ycc[4];
        toYCC(rgb + 3 * i, ycc);
        ch[0][i] = ycc[0]; ch[1][i] = ycc[1];
    }
    double num = 0.0, den = 0.0;
    const double cx = 0.5 * (w - 1), cy = 0.5 * (h - 1);
    for (std::vector<double>& v : ch) {
        // least-squares plane a + b x + c y (centred coordinates: the normal equations decouple)
        double s = 0, sx = 0, sy = 0, xx = 0, yy = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const double val = v[(size_t)y * w + x], dx = x - cx, dy = y - cy;
                s += val; sx += val * dx; sy += val * dy; xx += dx * dx; yy += dy * dy;
            }
        const double a = s / (double)N, b = xx > 0 ? sx / xx : 0, c = yy > 0 ? sy / yy : 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) v[(size_t)y * w + x] -= a + b * (x - cx) + c * (y - cy);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const double q = v[(size_t)y * w + x];
                den += q * q;
                if (x + 1 < w) num += 0.5 * q * v[(size_t)y * w + x + 1];
                if (y + 1 < h) num += 0.5 * q * v[(size_t)(y + 1) * w + x];
            }
    }
    if (den <= 1e-20) return 0.0;
    const double rho = std::min(0.98, num / den);
    const double g = rho > 0.01 ? std::sqrt(-1.0 / (4.0 * std::log(rho))) : 0.0;
    return std::min(50.0, std::max(2.0, std::round(40.0 * g) / 10.0));
}

} // namespace mk
