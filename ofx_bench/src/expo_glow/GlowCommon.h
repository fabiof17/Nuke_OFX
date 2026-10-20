// GlowCommon.h
// Per-pixel / per-line maths shared by the CPU and CUDA back-ends of apGlow_OFX.
#pragma once

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#define EG_HD __host__ __device__ inline
#else
#define EG_HD inline
#endif

namespace expoglow {

// Source preparation (luminance response).
struct PrepParams {
    int   width  = 0;      // working buffer size
    int   height = 0;
    float black  = 0.f;    // src black
    float white  = 1.f;    // src white
    bool  whiteClamp = false;
    float gamma  = 1.f;    // response
};

constexpr int kMaxTaps = 49;   // longest merged kernel (24 px each side)

// One blur step of the exponential glow: a cascade of separable box filters
// (fractional radius, zero outside the image) and the RGB weight of the step.
struct StepPlan {
    int   factor = 1;          // > 1: blur computed on an image reduced by this factor (Nuke "quality")
    int   passes = 0;          // 0 = no blur (size too small)
    float radius[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    float weight[3] = { 0.f, 0.f, 0.f };
    // Small cascades are merged into one equivalent 1D kernel (same weights, 2 passes
    // instead of 2 x passes). taps = 0 : use the box cascade.
    int   taps = 0;
    float kernel[kMaxTaps];
};

EG_HD float egFinite(float v) { return (v == v && fabsf(v) < 3.0e38f) ? v : 0.f; }

// Nuke Grade with black point / white point / gamma, black clamp on, optional white clamp.
EG_HD float gradeLuminance(const PrepParams& p, float lum)
{
    const float d = p.white - p.black;
    float x = fabsf(d) > 1e-12f ? (lum - p.black) / d : (lum >= p.black ? 1.f : 0.f);
    if (x <= 0.f) return 0.f;                       // black clamp
    if (p.gamma != 1.f) x = p.gamma > 1e-6f ? powf(x, 1.f / p.gamma) : (x >= 1.f ? x : 0.f);
    if (p.whiteClamp && x > 1.f) x = 1.f;
    return x;
}

// Emitting image: source (x area) re-scaled so that its luminance follows the response curve.
// out = rgb * grade(lum) / lum   (lum = Rec.709 luminance)
EG_HD void prepPixel(const PrepParams& p, const float* src4, float area, float* out4)
{
    float r = egFinite(src4[0]) * area, g = egFinite(src4[1]) * area, b = egFinite(src4[2]) * area;
    const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    const float k = lum > 1e-20f ? gradeLuminance(p, lum) / lum : 0.f;
    out4[0] = r * k; out4[1] = g * k; out4[2] = b * k; out4[3] = 0.f;
}

// Box filter of fractional radius r along one line of `n` RGBA pixels, `stride` floats apart.
// Weights: 1 for |k| <= floor(r), (r - floor(r)) at +/-(floor(r)+1), normalised by 2r+1.
// Outside the line the edge pixels repeat, like Nuke outside a bounding box.
EG_HD int egClampi(int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

EG_HD void boxLine(const float* in, float* out, int n, size_t stride, float r)
{
    const int   m = (int)floorf(r);
    const float a = r - (float)m;
    const double norm = 1.0 / (2.0 * (double)r + 1.0);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int j = -m; j <= m; ++j) {
        const float* q = in + (size_t)egClampi(j, n) * stride;
        s0 += q[0]; s1 += q[1]; s2 += q[2];
    }
    for (int i = 0; i < n; ++i) {
        const float* ql = in + (size_t)egClampi(i - m - 1, n) * stride;
        const float* qr = in + (size_t)egClampi(i + m + 1, n) * stride;
        float* o = out + (size_t)i * stride;
        o[0] = (float)((s0 + a * ((double)ql[0] + qr[0])) * norm);
        o[1] = (float)((s1 + a * ((double)ql[1] + qr[1])) * norm);
        o[2] = (float)((s2 + a * ((double)ql[2] + qr[2])) * norm);
        o[3] = 0.f;
        const float* qo = in + (size_t)egClampi(i - m, n) * stride;
        s0 += (double)qr[0] - qo[0]; s1 += (double)qr[1] - qo[1]; s2 += (double)qr[2] - qo[2];
    }
}

// Area-average downscale by an integer factor f: small pixel (sx, sy) = mean of its f x f block
// (blocks cut by the image edge average the pixels that exist).
EG_HD void downsamplePixel(const float* in, int w, int h, int f, int sx, int sy, float* out4)
{
    const int x0 = sx * f, y0 = sy * f;
    const int x1 = x0 + f < w ? x0 + f : w, y1 = y0 + f < h ? y0 + f : h;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int y = y0; y < y1; ++y) {
        const float* q = in + 4 * ((size_t)y * w + x0);
        for (int x = x0; x < x1; ++x, q += 4) { s0 += q[0]; s1 += q[1]; s2 += q[2]; }
    }
    const double k = 1.0 / ((double)(x1 - x0) * (double)(y1 - y0));
    out4[0] = (float)(s0 * k); out4[1] = (float)(s1 * k); out4[2] = (float)(s2 * k); out4[3] = 0.f;
}

// Bilinear upscale of a reduced image (factor f) at full-resolution pixel (x, y), edges clamped,
// accumulated into acc with the step weight.
EG_HD void upsampleAccumulate(const float* small, int ws, int hs, int f, int x, int y, float* acc4, const float* w3)
{
    float u = ((float)x + 0.5f) / (float)f - 0.5f;
    float v = ((float)y + 0.5f) / (float)f - 0.5f;
    u = u < 0.f ? 0.f : (u > (float)(ws - 1) ? (float)(ws - 1) : u);
    v = v < 0.f ? 0.f : (v > (float)(hs - 1) ? (float)(hs - 1) : v);
    const int u0 = (int)u, v0 = (int)v;
    const int u1 = u0 + 1 < ws ? u0 + 1 : u0, v1 = v0 + 1 < hs ? v0 + 1 : v0;
    const float fu = u - (float)u0, fv = v - (float)v0;
    const float* a = small + 4 * ((size_t)v0 * ws + u0);
    const float* b = small + 4 * ((size_t)v0 * ws + u1);
    const float* c = small + 4 * ((size_t)v1 * ws + u0);
    const float* d = small + 4 * ((size_t)v1 * ws + u1);
    for (int k = 0; k < 3; ++k) {
        // vertical first, then horizontal (same order as the CPU row-by-row version)
        const float left  = a[k] + (c[k] - a[k]) * fv;
        const float right = b[k] + (d[k] - b[k]) * fv;
        const float val = left + (right - left) * fu;
        const float add = val * w3[k];
        acc4[k] += (add == add && fabsf(add) < 3.0e38f) ? add : 0.f;
    }
}

// 1D convolution with the merged kernel (centre = taps/2), `stride` floats between pixels.
// Indices are clamped, which is exact on an image padded with its repeated edges.
EG_HD void convPixel(const float* in, int n, size_t stride, int i, const float* k, int taps, float* out4)
{
    const int c = taps / 2;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    for (int t = 0; t < taps; ++t) {
        const float* q = in + (size_t)egClampi(i + t - c, n) * stride;
        s0 += k[t] * q[0]; s1 += k[t] * q[1]; s2 += k[t] * q[2];
    }
    out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = 0.f;
}

// Copy of `in` (w x h) into a buffer padded by `pad` pixels on every side, the edge pixels
// being repeated (exact extension of the image as Nuke sees it beyond its bbox).
EG_HD void padPixel(const float* in, int w, int h, int pad, int px, int py, float* out4)
{
    const int x = egClampi(px - pad, w), y = egClampi(py - pad, h);
    const float* q = in + 4 * ((size_t)y * w + x);
    out4[0] = q[0]; out4[1] = q[1]; out4[2] = q[2]; out4[3] = 0.f;
}

// accum += weight * blurred (NaN-safe)
EG_HD void accumulatePixel(float* acc4, const float* blur4, const float* w3)
{
    acc4[0] += egFinite(blur4[0] * w3[0]);
    acc4[1] += egFinite(blur4[1] * w3[1]);
    acc4[2] += egFinite(blur4[2] * w3[2]);
}

} // namespace expoglow
