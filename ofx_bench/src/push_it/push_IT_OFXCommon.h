// push_IT_OFXCommon.h
// Per-pixel maths shared by the CPU and CUDA back-ends.
// Every function here is compiled for both host and device so that the two
// paths produce the same image (up to float rounding).
#pragma once

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#define PS_HD __host__ __device__ inline
#else
#define PS_HD inline
#endif

namespace pushit {

enum SpreadType   { kTypeInterpolate = 0, kTypeStretch = 1, kTypeVectorWarp = 2 };
enum SpreadMode   { kModeExpand = 0, kModeContract = 1 };
enum VectorSource { kVectorInput = 0, kVectorConstant = 1 };
enum ResultOutput { kResultCombined = 0, kResultSpreadOnly = 1, kResultPremult = 2 };
enum AlphaOutput  { kAlphaInput = 0, kAlphaSpreadMatte = 1, kAlphaSolid = 2 };

// All distances are in pixels of the image being rendered
// (the plug-in layer already multiplied them by the render scale).
struct Params {
    int   width  = 0;
    int   height = 0;

    int   spreadType = kTypeInterpolate;
    int   spreadMode = kModeExpand;
    bool  invertMatte = false;
    bool  sourcePremultiplied = true;

    float shrink        = 0.f;   // erosion of the valid area (px)
    float edgeSoftness  = 0.f;   // blur of the valid area (px)
    float seedThreshold = 0.5f;  // valid weight above which a pixel can be copied (Stretch / distance)
    float maxDistance   = 0.f;   // 0 = unlimited
    float falloff       = 0.f;   // soft end of the spread (px)
    float blur          = 0.f;   // blur of the spread pixels (px)

    // Vector Warp
    int   vectorSource    = kVectorInput;
    float constX = 1.f, constY = 0.f;   // constant spread vector (px)
    float motionDistance  = 1.f;        // gain applied to the vector input
    float motionThreshold = 0.f;        // vectors shorter than this (px) are ignored
    float originX = 0.f, originY = 0.f;
    float gainRX = 1.f, gainRY = 0.f;
    float gainGX = 0.f, gainGY = 1.f;
    float gainBX = 0.f, gainBY = 0.f;
    bool  invertDirection = false;
    int   maxMarchSteps   = 4096;

    // Output
    int   resultOutput = kResultCombined;
    int   alphaOutput  = kAlphaInput;
    bool  premultiplyOutput = false;
};

struct Seed { int x, y; };   // x < 0 : no seed

// ---------------------------------------------------------------- helpers
PS_HD float psClamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
PS_HD int   psClampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

// A blur "size" in px is approximated by three successive box filters.
// Three boxes of radius r have a variance of r*(r+1); we match sigma = size/2.
PS_HD int boxRadiusForBlur(float size)
{
    if (size <= 0.f) return 0;
    const float sigma = 0.5f * size;
    int r = (int)floorf(sqrtf(sigma * sigma + 0.25f) - 0.5f + 0.5f);
    return r < 1 ? 1 : r;
}

// ------------------------------------------------------------ valid area
// "valid" = pixels that are kept and used as the colour source of the spread.
// Expand  : valid = matte       -> colours are pushed outward from the matte.
// Contract: valid = 1 - matte   -> the matted area is filled from its surroundings.
PS_HD float computeValid(const Params& p, float matte)
{
    float m = psClamp01(matte);
    if (p.invertMatte) m = 1.f - m;
    return p.spreadMode == kModeExpand ? m : 1.f - m;
}

// src4 = input pixel as delivered by the host, out4 = rgb (unpremultiplied if needed) + source alpha
PS_HD void unpremultSource(const Params& p, const float* src4, float* out4)
{
    const float a = src4[3];
    // Only meaningful in Expand: the kept pixels are the foreground of a premultiplied
    // image. In Contract the kept pixels are the background, unpremultiplying them
    // would bring foreground colours back from the transparent edge pixels.
    if (p.sourcePremultiplied && p.spreadMode == kModeExpand && a > 1e-6f) {
        const float ia = 1.f / a;
        out4[0] = src4[0] * ia; out4[1] = src4[1] * ia; out4[2] = src4[2] * ia;
    } else {
        out4[0] = src4[0]; out4[1] = src4[1]; out4[2] = src4[2];
    }
    out4[3] = a;
}

// base = colour weighted by validity (P) + weight (W)
PS_HD void buildBase(const float* rgbU4, float v, float* base4)
{
    base4[0] = rgbU4[0] * v; base4[1] = rgbU4[1] * v; base4[2] = rgbU4[2] * v; base4[3] = v;
}

// ------------------------------------------------------- push-pull (Interpolate)
// Pull: 2x2 sum, saturated so that the weight never exceeds 1.
PS_HD void pullPixel(const float* fine, int fw, int fh, int cx, int cy, float* out4)
{
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int j = 0; j < 2; ++j) {
        const int fy = 2 * cy + j;
        if (fy >= fh) break;
        for (int i = 0; i < 2; ++i) {
            const int fx = 2 * cx + i;
            if (fx >= fw) break;
            const float* f = fine + 4 * ((size_t)fy * fw + fx);
            s0 += f[0]; s1 += f[1]; s2 += f[2]; s3 += f[3];
        }
    }
    if (s3 > 1.f) { const float k = 1.f / s3; s0 *= k; s1 *= k; s2 *= k; s3 = 1.f; }
    out4[0] = s0; out4[1] = s1; out4[2] = s2; out4[3] = s3;
}

// Bilinear fetch, clamped to the edges.
PS_HD void bilinearClamp4(const float* img, int w, int h, float x, float y, float* o)
{
    x = x < 0.f ? 0.f : (x > (float)(w - 1) ? (float)(w - 1) : x);
    y = y < 0.f ? 0.f : (y > (float)(h - 1) ? (float)(h - 1) : y);
    const int x0 = (int)x, y0 = (int)y;
    const int x1 = x0 + 1 < w ? x0 + 1 : x0;
    const int y1 = y0 + 1 < h ? y0 + 1 : y0;
    const float fx = x - (float)x0, fy = y - (float)y0;
    const float* a = img + 4 * ((size_t)y0 * w + x0);
    const float* b = img + 4 * ((size_t)y0 * w + x1);
    const float* c = img + 4 * ((size_t)y1 * w + x0);
    const float* d = img + 4 * ((size_t)y1 * w + x1);
    for (int k = 0; k < 4; ++k) {
        const float top = a[k] + (b[k] - a[k]) * fx;
        const float bot = c[k] + (d[k] - c[k]) * fx;
        o[k] = top + (bot - top) * fy;
    }
}

// Push: fill what is missing at this level with the (already filled) coarser level.
PS_HD void pushPixel(float* fine4, const float* coarse, int cw, int ch, int fx, int fy)
{
    float c[4];
    bilinearClamp4(coarse, cw, ch, ((float)fx + 0.5f) * 0.5f - 0.5f, ((float)fy + 0.5f) * 0.5f - 0.5f, c);
    const float k = 1.f - psClamp01(fine4[3]);
    fine4[0] += k * c[0]; fine4[1] += k * c[1]; fine4[2] += k * c[2]; fine4[3] += k * c[3];
}

// Turns a pushed P/W pixel into a colour. Returns false if nothing could be found.
PS_HD bool resolveWeighted(const float* pw4, float* rgb3)
{
    if (pw4[3] > 1e-12f) {
        const float k = 1.f / pw4[3];
        rgb3[0] = pw4[0] * k; rgb3[1] = pw4[1] * k; rgb3[2] = pw4[2] * k;
        return true;
    }
    rgb3[0] = rgb3[1] = rgb3[2] = 0.f;
    return false;
}

PS_HD long long seedDist2(Seed s, int x, int y)
{
    const long long dx = (long long)(s.x - x), dy = (long long)(s.y - y);
    return dx * dx + dy * dy;
}

// ------------------------------------- exact distance transform (Stretch, distances)
// Felzenszwalb & Huttenlocher separable squared Euclidean distance transform, keeping the
// nearest seed. Exact, two passes over the image (replaces jump flooding).
PS_HD bool isSeed(float v, float threshold) { return v > 0.f && v >= threshold; }

// Pass 1, one column x: nearestY[y*w+x] = row of the nearest seed in this column, or -1.
// Ties go to the upper seed (smaller y).
PS_HD void edtColumn(const float* valid, int w, int h, int x, float thr, int* nearestY)
{
    int last = -1;
    for (int y = 0; y < h; ++y) {
        if (isSeed(valid[(size_t)y * w + x], thr)) last = y;
        nearestY[(size_t)y * w + x] = last;
    }
    int next = -1;
    for (int y = h - 1; y >= 0; --y) {
        if (isSeed(valid[(size_t)y * w + x], thr)) next = y;
        const int up = nearestY[(size_t)y * w + x];
        int best = up;
        if (up < 0) best = next;
        else if (next >= 0 && next - y < y - up) best = next;
        nearestY[(size_t)y * w + x] = best;
    }
}

// Pass 2, one row y: lower envelope of the parabolas (x - q)^2 + (y - nearestY(q, y))^2.
// Exact integer arithmetic: every parabola intersection is the fraction zn / zd (zd > 0),
// compared by cross-multiplication in 64 bits (no rounding, no slow double division on
// consumer GPUs, and the CPU and the GPU take exactly the same decisions).
// Element i of the row lives at ny[i * s], v[i * s], zn[i * s], zd[i * s] and out[i * s]:
// the CPU uses s = 1, the GPU interleaves the rows (s = image height) so reads coalesce.
// v, zn, zd are scratch arrays of w elements.
PS_HD void edtRow(const int* ny, int w, int y, size_t s, int* v, long long* zn, int* zd, Seed* out)
{
    int k = -1;
    for (int q = 0; q < w; ++q) {
        const int nq = ny[(size_t)q * s];
        if (nq < 0) continue;
        const long long dq = y - nq;
        const long long fq = dq * dq + (long long)q * q;
        while (true) {
            if (k < 0) { k = 0; v[0] = q; break; }
            const int p = v[(size_t)k * s];
            const long long dp = y - ny[(size_t)p * s];
            const long long num = fq - (dp * dp + (long long)p * p);
            const long long den = 2LL * (q - p);                 // > 0
            // intersection num/den <= z[k] ?  (z[0] is -infinity)
            if (k > 0 && num * (long long)zd[(size_t)k * s] <= zn[(size_t)k * s] * den) { --k; continue; }
            ++k; v[(size_t)k * s] = q; zn[(size_t)k * s] = num; zd[(size_t)k * s] = (int)den;
            break;
        }
    }
    if (k < 0) {
        for (int x = 0; x < w; ++x) { out[(size_t)x * s].x = -1; out[(size_t)x * s].y = -1; }
        return;
    }
    int j = 0;
    for (int x = 0; x < w; ++x) {
        while (j < k && zn[(size_t)(j + 1) * s] < (long long)x * zd[(size_t)(j + 1) * s]) ++j;   // z[k+1] = +infinity
        const int vj = v[(size_t)j * s];
        out[(size_t)x * s].x = vj; out[(size_t)x * s].y = ny[(size_t)vj * s];
    }
}

// ------------------------------------------------- sliding box (blur, edge softness)
// Box of radius r along one series of n pixels, `stride` floats apart, on the first `nch`
// channels (1..4) at once, edges clamped, double accumulators. Same arithmetic on CPU and GPU.
PS_HD void slideBox(const float* in, float* out, int n, size_t stride, int nch, int r)
{
    const double norm = 1.0 / (double)(2 * r + 1);
    double sum[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = -r; i <= r; ++i) {
        const float* q = in + (size_t)psClampi(i, 0, n - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += q[c];
    }
    for (int x = 0; x < n; ++x) {
        float* o = out + (size_t)x * stride;
        for (int c = 0; c < nch; ++c) o[c] = (float)(sum[c] * norm);
        const float* qa = in + (size_t)psClampi(x + r + 1, 0, n - 1) * stride;
        const float* qs = in + (size_t)psClampi(x - r, 0, n - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += (double)qa[c] - (double)qs[c];
    }
}

// ----------------------------------------------------- vector warp
PS_HD void bilinearZero4(const float* img, int w, int h, float x, float y, float* o)
{
    o[0] = o[1] = o[2] = o[3] = 0.f;
    const float fx0 = floorf(x), fy0 = floorf(y);
    const int x0 = (int)fx0, y0 = (int)fy0;
    const float fx = x - fx0, fy = y - fy0;
    const float wts[4] = { (1.f - fx) * (1.f - fy), fx * (1.f - fy), (1.f - fx) * fy, fx * fy };
    const int xs[4] = { x0, x0 + 1, x0, x0 + 1 };
    const int ys[4] = { y0, y0, y0 + 1, y0 + 1 };
    for (int n = 0; n < 4; ++n) {
        if (xs[n] < 0 || ys[n] < 0 || xs[n] >= w || ys[n] >= h || wts[n] == 0.f) continue;
        const float* s = img + 4 * ((size_t)ys[n] * w + xs[n]);
        o[0] += wts[n] * s[0]; o[1] += wts[n] * s[1]; o[2] += wts[n] * s[2]; o[3] += wts[n] * s[3];
    }
}

// Direction in which pixels are spread at this location (px).
PS_HD void spreadVector(const Params& p, const float* vec4, float* dx, float* dy)
{
    float x, y;
    if (p.vectorSource == kVectorConstant) {
        x = p.constX; y = p.constY;
    } else if (vec4) {
        const float r = vec4[0] - p.originX;
        const float g = vec4[1] - p.originY;
        const float b = vec4[2];
        x = (p.gainRX * r + p.gainGX * g + p.gainBX * b) * p.motionDistance;
        y = (p.gainRY * r + p.gainGY * g + p.gainBY * b) * p.motionDistance;
    } else {
        x = 0.f; y = 0.f;
    }
    if (p.invertDirection) { x = -x; y = -y; }
    *dx = x; *dy = y;
}

// Lower bound of the distance from (qx, qy) to the nearest pixel with a non-zero weight,
// read from the jump-flooding distance field (all seeds live inside the image).
PS_HD float distanceLowerBound(const float* distField, int w, int h, float qx, float qy)
{
    const float cx = qx < 0.f ? 0.f : (qx > (float)(w - 1) ? (float)(w - 1) : qx);
    const float cy = qy < 0.f ? 0.f : (qy > (float)(h - 1) ? (float)(h - 1) : qy);
    const float ox = qx - cx, oy = qy - cy;
    const float outside = sqrtf(ox * ox + oy * oy);
    const int ix = (int)(cx + 0.5f), iy = (int)(cy + 0.5f);
    const float inside = distField[(size_t)iy * w + ix] - 0.7072f - outside;
    return outside > inside ? outside : inside;
}

// Walks backward along the spread vector and gathers the first valid pixels met
// (front-to-back accumulation). Empty stretches are skipped with the distance
// field (sphere tracing), so long vectors stay cheap. Returns the coverage, colour in C.
PS_HD float marchPixel(const Params& p, const float* base, const float* distField, int w, int h,
                       int x, int y, float dx, float dy, float* C)
{
    C[0] = C[1] = C[2] = 0.f;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len < 0.5f || len < p.motionThreshold) return 0.f;
    const float ux = dx / len, uy = dy / len;
    float A = 0.f;
    float s = 1.f;
    for (int it = 0; it < p.maxMarchSteps && s <= len; ++it) {
        const float qx = (float)x - ux * s, qy = (float)y - uy * s;
        const float d = distanceLowerBound(distField, w, h, qx, qy);
        if (d > 2.42f) {                // bilinear footprint is empty: jump
            s += floorf(d - 1.42f);     // whole pixels, so s stays an exact integer: the
            continue;                   // CPU and the GPU then sample exactly the same points
        }
        float smp[4];
        bilinearZero4(base, w, h, qx, qy, smp);
        s += 1.f;
        if (smp[3] <= 0.f) continue;
        const float k = 1.f - A;
        C[0] += k * smp[0]; C[1] += k * smp[1]; C[2] += k * smp[2];
        A += k * smp[3];
        if (A > 0.999f) break;
    }
    return A;
}

// Distance (px) to the nearest seed, or a huge value when there is none.
PS_HD float seedDistance(Seed s, int x, int y)
{
    return s.x >= 0 ? sqrtf((float)seedDist2(s, x, y)) : 1e30f;
}

PS_HD float effectiveSeedThreshold(const Params& p)
{
    // Vector Warp uses the distance field to skip empty space: every non-zero pixel must be a seed.
    return p.spreadType == kTypeVectorWarp ? 0.f : p.seedThreshold;
}

// ----------------------------------------------------------- spread matte
PS_HD float spreadMatte(const Params& p, bool hasFill, bool hasSeed, float dist)
{
    if (!hasFill) return 0.f;
    if (p.maxDistance <= 0.f) return 1.f;
    if (!hasSeed) return 0.f;
    const float f = p.falloff > 0.f ? p.falloff : 0.f;
    const float inner = p.maxDistance - f;
    if (dist <= inner) return 1.f;
    if (dist >= p.maxDistance) return 0.f;
    float t = (p.maxDistance - dist) / (f > 1e-6f ? f : 1e-6f);   // 1 at inner, 0 at max
    return t * t * (3.f - 2.f * t);                                 // smoothstep
}

// ------------------------------------------------------------- composite
PS_HD void compositePixel(const Params& p, const float* rgbU4, float v,
                          const float* fill3, float sm, float* out4)
{
    float r = fill3[0] * sm, g = fill3[1] * sm, b = fill3[2] * sm;
    const bool combined = p.resultOutput != kResultSpreadOnly;   // Combined and Premult
    if (combined) {
        const float iv = 1.f - v;
        r = rgbU4[0] * v + r * iv;
        g = rgbU4[1] * v + g * iv;
        b = rgbU4[2] * v + b * iv;
    }
    float a;
    if (p.alphaOutput == kAlphaInput)
        a = rgbU4[3];
    else if (p.alphaOutput == kAlphaSpreadMatte)
        a = combined ? v + (1.f - v) * sm : sm;
    else
        a = 1.f;
    if (p.premultiplyOutput || p.resultOutput == kResultPremult) { r *= a; g *= a; b *= a; }
    out4[0] = r; out4[1] = g; out4[2] = b; out4[3] = a;
}

} // namespace pushit
