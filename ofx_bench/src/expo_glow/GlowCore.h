// GlowCore.h
// Glow planning (host side) and back-end entry points.
// Buffers are dense RGBA float images, row-major, width * height pixels.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "GlowCommon.h"

namespace expoglow {

enum BlurFilter { kFilterBox = 0, kFilterTriangle = 1, kFilterQuadratic = 2, kFilterGaussian = 3 };

// Box / triangle / quadratic, measured in Nuke (1 white pixel, Blur size 100):
// each filter keeps its natural width (box 1, triangle 2, quadratic 3) times the size,
// i.e. a cascade of 1, 2 or 3 boxes whose width equals the size.
// Gaussian: measured in Nuke (1 white pixel, Blur gaussian size 100 -> centre 0.00009),
// which matches size = FWHM, i.e. sigma = size / (2 * sqrt(2 ln 2)) = 0.4247 * size.
constexpr double kGaussianSigmaPerSize = 0.42466090014400953;

struct GlowSettings {
    float intensity = 1.f;
    float persistence[3] = { 1.f, 1.f, 1.f };
    float tint[3] = { 1.f, 1.f, 1.f };
    float maxSize = 2000.f;     // px at full resolution
    float sizeFalloff = 1.f;
    float sizeScale = 1.f;      // fine calibration of every blur size
    float quality = 15.f;       // Nuke Blur "quality" (0 = exact filter)
    int   steps = 10;
    int   filter = kFilterGaussian;
    float renderScale = 1.f;
};

// Variance of the fractional box used by boxLine().
inline double fractionalBoxVariance(double r)
{
    const double m = std::floor(r), a = r - m;
    return (m * (m + 1.0) * (2.0 * m + 1.0) / 3.0 + 2.0 * a * (m + 1.0) * (m + 1.0)) / (2.0 * r + 1.0);
}

// Radius of the fractional box whose variance is `var` (bisection, monotonic).
inline double boxRadiusForVariance(double var)
{
    if (var <= 0.0) return 0.0;
    double lo = 0.0, hi = std::sqrt(3.0 * var) + 2.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (fractionalBoxVariance(mid) < var ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

// Box cascade reproducing a filter of the given size (px).
// Nuke Blur "quality": when the size exceeds the quality, Nuke scales the image down,
// filters it, then scales it back up with linear interpolation. We do the same: the blur
// runs on an image reduced by an integer factor f (area average), and is scaled back up
// bilinearly. That is much faster for large blurs and softens the filter edges like Nuke.
// The factor is calibrated on a Nuke render (box filter, edges 10-90 % about 6.5 px for a
// 45 px blur at quality 15): f = 1.5 * size / quality.
constexpr double kQualitySoftness = 1.5;

inline StepPlan planBlur(int filter, double size, double quality = 0.0)
{
    StepPlan s;
    if (size <= 1e-3) return s;
    if (quality > 0.0 && size > quality) {
        s.factor = std::max(2, (int)std::lround(kQualitySoftness * size / quality));
        size /= s.factor;                    // filter size on the reduced image
    }
    auto boxR = [](double width) { return std::max(0.0, (width - 1.0) * 0.5); };
    switch (filter) {
    case kFilterBox:       s.passes = 1; s.radius[0] = (float)boxR(size); break;
    case kFilterTriangle:  s.passes = 2; for (int i = 0; i < 2; ++i) s.radius[i] = (float)boxR(size); break;
    case kFilterQuadratic: s.passes = 3; for (int i = 0; i < 3; ++i) s.radius[i] = (float)boxR(size); break;
    default: {
        const double sigma = size * kGaussianSigmaPerSize;
        s.passes = 4;
        const float r = (float)boxRadiusForVariance(sigma * sigma / 4.0);
        for (int i = 0; i < 4; ++i) s.radius[i] = r;
    }
    }
    bool any = false;
    for (int i = 0; i < s.passes; ++i) any = any || s.radius[i] > 1e-4f;
    if (!any) s.passes = 0;
    return s;
}

// Size of step i (1..steps): (maxSize * falloff)^(i/steps) / falloff
inline double stepSize(const GlowSettings& g, int i)
{
    const double base = (double)g.maxSize * (double)g.sizeFalloff;
    if (base <= 0.0 || g.sizeFalloff <= 0.f || g.steps <= 0) return 0.0;
    return std::pow(base, (double)i / g.steps) / g.sizeFalloff * g.sizeScale * g.renderScale;
}

// Weights of the fractional box used by boxLine(), as a discrete kernel.
inline std::vector<double> fractionalBoxKernel(double r)
{
    const int m = (int)std::floor(r);
    const double a = r - m, norm = 1.0 / (2.0 * r + 1.0);
    std::vector<double> k(2 * m + 3, norm);
    k.front() = k.back() = a * norm;
    return k;
}

// Merges a short cascade into one kernel (convolution of its boxes).
inline void mergeKernel(StepPlan& s)
{
    s.taps = 0;
    if (s.passes == 0) return;
    std::vector<double> k(1, 1.0);
    for (int i = 0; i < s.passes; ++i) {
        const std::vector<double> b = fractionalBoxKernel(s.radius[i]);
        std::vector<double> c(k.size() + b.size() - 1, 0.0);
        for (size_t x = 0; x < k.size(); ++x)
            for (size_t y = 0; y < b.size(); ++y) c[x + y] += k[x] * b[y];
        k.swap(c);
    }
    // trim the negligible ends (< 1e-4 of the peak, invisible) and renormalise
    double peak = 0.0;
    for (double v : k) peak = std::max(peak, v);
    size_t lo = 0, hi = k.size();
    while (hi - lo > 1 && k[lo] < 1e-4 * peak && k[hi - 1] < 1e-4 * peak) { ++lo; --hi; }
    double sum = 0.0;
    for (size_t t = lo; t < hi; ++t) sum += k[t];
    for (size_t t = lo; t < hi; ++t) k[t] /= sum;
    if (hi - lo > (size_t)kMaxTaps) return;          // too long: keep the box cascade
    s.taps = (int)(hi - lo);
    for (int t = 0; t < s.taps; ++t) s.kernel[t] = (float)k[lo + t];
}

inline std::vector<StepPlan> planGlow(const GlowSettings& g)
{
    std::vector<StepPlan> plan;
    const int s = std::max(1, g.steps);
    for (int i = 1; i <= s; ++i) {
        StepPlan p = planBlur(g.filter, stepSize(g, i), g.quality);
        mergeKernel(p);
        const float t = (float)i / (float)s;
        for (int c = 0; c < 3; ++c)
            p.weight[c] = (g.persistence[c] * t + (1.f - t)) * g.intensity * g.tint[c] / (float)s;
        plan.push_back(p);
    }
    return plan;
}

// Margin (px of the blurred image) after which a cascade no longer sees the image edges.
inline int planPadding(const StepPlan& p)
{
    int e = 0;
    for (int i = 0; i < p.passes; ++i) e += (int)std::floor(p.radius[i]) + 1;
    return e;
}

// How far (px) the glow reaches beyond the emitting pixels.
inline int planExtent(const std::vector<StepPlan>& plan)
{
    int ext = 0;
    for (const StepPlan& p : plan) {
        int e = 0;
        for (int i = 0; i < p.passes; ++i) e += (int)std::floor(p.radius[i]) + 1;
        if (p.factor > 1) e = (e + 2) * p.factor;   // reduced pixels + resampling footprint
        ext = std::max(ext, e);
    }
    return ext;
}

// glowRGBA receives the glow only (alpha = 0). `area` (1 float / px) may be null.
void computeGlowCPU(const PrepParams& p, const std::vector<StepPlan>& plan,
                    const float* srcRGBA, const float* area, float* glowRGBA);

bool cudaIsCompiledIn();
bool cudaDeviceInfo(std::string& description);
bool computeGlowCUDA(const PrepParams& p, const std::vector<StepPlan>& plan,
                     const float* srcRGBA, const float* area, float* glowRGBA, std::string& error);
// Same, streaming rows straight from / into the transfer memory (no intermediate copy of the
// whole image, and only what is needed crosses the bus):
//   srcRGB(y0, y1, rows)  writes source rows [y0, y1), p.width pixels of 3 floats (RGB)
//   area(y0, y1, rows)    writes area rows, 1 float per pixel (nullptr: no area)
//   glowRGB(y0, y1, rows) receives the glow of rows [y0, y1) of the window
//                         (outX, outY, outW, outH) of the buffer, 3 floats per pixel
// The callbacks are called from several threads at once, on disjoint rows.
using RowReader = std::function<void(int y0, int y1, float* rows)>;
using RowWriter = std::function<void(int y0, int y1, const float* rows)>;
bool computeGlowCUDARows(const PrepParams& p, const std::vector<StepPlan>& plan,
                         const RowReader& srcRGB, const RowReader* area,
                         int outX, int outY, int outW, int outH, const RowWriter& glowRGB, std::string& error);
void cudaReleaseResources();
// Per-stage GPU timings (benchmark tool only).
void cudaSetProfiling(bool on);
std::string cudaLastProfile();

} // namespace expoglow
