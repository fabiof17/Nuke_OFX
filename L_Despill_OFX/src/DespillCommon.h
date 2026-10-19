// DespillCommon.h
// Per-pixel maths of L_Despill_OFX, shared by the CPU and CUDA back-ends (compiled for
// host and device) so that both paths give the same image.
//
// Pipeline (all colours are scene-linear on input and output):
//   1. despill   : the screen channel is limited by a mix (average or max) of the two other
//                  channels, optionally in log (Cineon) space, around an optional balance
//                  (white / black point) of the screen channel. Only the screen channel changes.
//   2. restore   : the luminance removed with the spill can be added back.
//   3. spill matte: amount of spill removed, graded, then median / clean / blur (spatial,
//                  done by the back-ends) and clamped to [0, 1].
//   4. spill replace: grade + saturation of the despilled image inside the spill matte.
//   5. FG grade  : grade + saturation outside the spill matte.
//   6. background: optional BG added inside the spill matte (multiply + plus) or keymixed.
//   7. alpha, mask, mix.
#pragma once

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#define LD_HD __host__ __device__ inline
#else
#define LD_HD inline
#endif

namespace ldespill {

enum ScreenColor { kScreenGreen = 0, kScreenBlue = 1, kScreenRed = 2 };
enum DespillMath { kMathAverage = 0, kMathMax = 1 };
enum MatteSpace  { kMatteSameAsDespill = 0, kMatteLinear = 1, kMatteLog = 2 };
enum BGMode      { kBGOff = 0, kBGAdd = 1, kBGKeymix = 2 };
enum AlphaOut    { kAlphaSource = 0, kAlphaSpillMatte = 1 };

// Nuke Grade: blackpoint, whitepoint, lift, gain, multiply, offset, gamma (per channel).
struct Grade {
    float blackpoint[3] = { 0.f, 0.f, 0.f };
    float whitepoint[3] = { 1.f, 1.f, 1.f };
    float lift[3]       = { 0.f, 0.f, 0.f };
    float gain[3]       = { 1.f, 1.f, 1.f };
    float multiply[3]   = { 1.f, 1.f, 1.f };
    float offset[3]     = { 0.f, 0.f, 0.f };
    float gamma[3]      = { 1.f, 1.f, 1.f };
    bool  blackClamp    = false;
    bool  whiteClamp    = false;
    float saturation    = 1.f;          // applied after the grade (Rec. 709 luminance)
};

struct Params {
    // working buffer (source with its margin) and output window inside it
    int bufW = 0, bufH = 0;             // buffer size
    int winX = 0, winY = 0;             // window origin in the buffer
    int winW = 0, winH = 0;             // window size

    // despill
    int   screen = kScreenGreen;
    int   math = kMathAverage;
    float bias = 0.f;                   // -1: 1st other channel only, +1: 2nd other channel only
    float limitScale = 1.f;             // multiplies the limit (lower = stronger despill)
    bool  mathLog = true;               // despill computed in log (Cineon) space
    float balanceWhite = 1.f;           // screen channel balance (white / black point)
    float balanceBlack = 0.f;
    float restoreLuminance = 0.f;       // 0..1

    // spill matte
    int   matteSpace = kMatteLinear;
    float matteBlack = 0.f, matteWhite = 1.f, matteGamma = 1.f;
    int   medianRadius = 0;             // px
    int   cleanRadius = 0;              // px, > 0 fills holes (close), < 0 removes specks (open)
    float matteBlur = 0.f;              // px

    // spill replace / FG grade
    Grade replace;
    bool  replaceLog = true;
    Grade fg;

    // background
    int   bgMode = kBGAdd;
    bool  bgLog = true;
    Grade bg;
    bool  hasBG = false;

    // output
    int   alphaOut = kAlphaSource;
    float mix = 1.f;
};

// ------------------------------------------------------------------ helpers
LD_HD float ldClamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
LD_HD int   ldClampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
LD_HD float ldLuma(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }

// Channel order for the screen colour: screen channel, 1st and 2nd other channels.
//   green: G, (R, B)   blue: B, (R, G)   red: R, (G, B)
LD_HD void screenChannels(int screen, int* s, int* o1, int* o2)
{
    if (screen == kScreenBlue)     { *s = 2; *o1 = 0; *o2 = 1; }
    else if (screen == kScreenRed) { *s = 0; *o1 = 1; *o2 = 2; }
    else                           { *s = 1; *o1 = 0; *o2 = 2; }
}

// Cineon log, as Nuke's Log2Lin node with black 0, white 685, gamma 0.6 (the settings
// used by L_Despill). Values below the representable range are clamped instead of
// producing NaN.
#define LD_LOG_OFFSET 0.0052077983f       // 10^((0 - 685) * 0.002 / 0.6)
LD_HD float lin2log(float x)
{
    float v = x * (1.f - LD_LOG_OFFSET) + LD_LOG_OFFSET;
    v = v > 1e-10f ? v : 1e-10f;
    return (685.f + log10f(v) * (0.6f / 0.002f)) / 1023.f;
}
LD_HD float log2lin(float x)
{
    return (powf(10.f, (1023.f * x - 685.f) * (0.002f / 0.6f)) - LD_LOG_OFFSET) / (1.f - LD_LOG_OFFSET);
}

// Nuke Grade + Saturation on one RGB colour.
LD_HD void applyGrade(const Grade& g, float* c)
{
    for (int k = 0; k < 3; ++k) {
        const float range = g.whitepoint[k] - g.blackpoint[k];
        const float A = g.multiply[k] * (g.gain[k] - g.lift[k]) / (fabsf(range) > 1e-8f ? range : 1e-8f);
        const float B = g.offset[k] + g.lift[k] - A * g.blackpoint[k];
        float y = A * c[k] + B;
        if (g.gamma[k] != 1.f && y > 0.f) y = powf(y, 1.f / (g.gamma[k] > 1e-4f ? g.gamma[k] : 1e-4f));
        if (g.blackClamp && y < 0.f) y = 0.f;
        if (g.whiteClamp && y > 1.f) y = 1.f;
        c[k] = y;
    }
    if (g.saturation != 1.f) {
        const float L = ldLuma(c);
        for (int k = 0; k < 3; ++k) c[k] = L + g.saturation * (c[k] - L);
    }
}

// Grade in log space when `inLog` (the colour stays linear on input and output).
LD_HD void applyGradeIn(const Grade& g, bool inLog, float* c)
{
    if (!inLog) { applyGrade(g, c); return; }
    for (int k = 0; k < 3; ++k) c[k] = lin2log(c[k]);
    applyGrade(g, c);
    for (int k = 0; k < 3; ++k) c[k] = log2lin(c[k]);
}

// Despill of a linear colour in linear or log space. Writes the despilled colour (linear)
// and returns the amount of spill removed, in the units of that space.
// Only the screen channel changes; the two other channels are copied as they are.
LD_HD float despillIn(const Params& p, bool inLog, const float* in, float* out)
{
    int s, o1, o2;
    screenChannels(p.screen, &s, &o1, &o2);
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2];

    // balance of the screen channel (removed again afterwards)
    const float range = p.balanceWhite - p.balanceBlack;
    const float br = fabsf(range) > 1e-6f ? range : 1e-6f;
    float vs = (in[s] - p.balanceBlack) / br;
    float v1 = in[o1], v2 = in[o2];
    if (inLog) { vs = lin2log(vs); v1 = lin2log(v1); v2 = lin2log(v2); }

    // limit from the two other channels; bias moves the weight from one to the other
    const float b = p.bias < -1.f ? -1.f : (p.bias > 1.f ? 1.f : p.bias);
    float limit;
    if (p.math == kMathMax) {
        const float w1 = 1.f - b < 1.f ? 1.f - b : 1.f, w2 = 1.f + b < 1.f ? 1.f + b : 1.f;
        const float a1 = w1 * v1, a2 = w2 * v2;
        limit = a1 > a2 ? a1 : a2;
    } else {
        limit = 0.5f * ((1.f - b) * v1 + (1.f + b) * v2);
    }
    limit *= p.limitScale;

    const float spill = vs > limit ? vs - limit : 0.f;
    if (spill > 0.f) {
        float d = vs - spill;
        if (inLog) d = log2lin(d);
        out[s] = d * br + p.balanceBlack;
    }
    return spill;
}

// Raw spill matte of one pixel (before the spatial filters): spill amount, graded.
LD_HD float matteRawPixel(const Params& p, const float* rgb)
{
    const bool inLog = p.matteSpace == kMatteSameAsDespill ? p.mathLog : p.matteSpace == kMatteLog;
    float tmp[3];
    float m = despillIn(p, inLog, rgb, tmp);
    const float range = p.matteWhite - p.matteBlack;
    m = (m - p.matteBlack) / (fabsf(range) > 1e-8f ? range : 1e-8f);
    if (m < 0.f) m = 0.f;
    if (p.matteGamma != 1.f && m > 0.f) m = powf(m, 1.f / (p.matteGamma > 1e-4f ? p.matteGamma : 1e-4f));
    return m;
}

// Final colour of one output pixel.
//   src4  : source RGBA (linear)         bg3 : background RGB or nullptr
//   matte : processed spill matte        mask: effect mask (1 when there is no mask)
LD_HD void finalPixel(const Params& p, const float* src4, const float* bg3, float matte, float mask, float* out4)
{
    const float M = ldClamp01(matte);
    float c[3];
    despillIn(p, p.mathLog, src4, c);

    // luminance removed with the spill, added back
    if (p.restoreLuminance != 0.f) {
        const float d[3] = { src4[0] - c[0] > 0.f ? src4[0] - c[0] : 0.f,
                             src4[1] - c[1] > 0.f ? src4[1] - c[1] : 0.f,
                             src4[2] - c[2] > 0.f ? src4[2] - c[2] : 0.f };
        const float L = p.restoreLuminance * ldLuma(d);
        c[0] += L; c[1] += L; c[2] += L;
    }

    // spill replace (inside the spill matte)
    if (M > 0.f) {
        float r[3] = { c[0], c[1], c[2] };
        applyGradeIn(p.replace, p.replaceLog, r);
        for (int k = 0; k < 3; ++k) c[k] += (r[k] - c[k]) * M;
    }
    // FG grade (outside the spill matte)
    if (M < 1.f) {
        float f[3] = { c[0], c[1], c[2] };
        applyGrade(p.fg, f);
        for (int k = 0; k < 3; ++k) c[k] += (f[k] - c[k]) * (1.f - M);
    }
    // background
    if (bg3 && p.hasBG && p.bgMode != kBGOff && M > 0.f) {
        if (p.bgMode == kBGKeymix) {
            for (int k = 0; k < 3; ++k) c[k] += (bg3[k] - c[k]) * M;
        } else if (p.bgLog) {           // multiply + plus in log space
            float g[3] = { lin2log(bg3[0]), lin2log(bg3[1]), lin2log(bg3[2]) };
            applyGrade(p.bg, g);
            for (int k = 0; k < 3; ++k) c[k] = log2lin(lin2log(c[k]) + g[k] * M);
        } else {
            float g[3] = { bg3[0], bg3[1], bg3[2] };
            applyGrade(p.bg, g);
            for (int k = 0; k < 3; ++k) c[k] += g[k] * M;
        }
    }

    const float a = p.alphaOut == kAlphaSpillMatte ? M : src4[3];
    const float k = p.mix * mask;
    out4[0] = src4[0] + (c[0] - src4[0]) * k;
    out4[1] = src4[1] + (c[1] - src4[1]) * k;
    out4[2] = src4[2] + (c[2] - src4[2]) * k;
    out4[3] = src4[3] + (a - src4[3]) * k;
}

// ------------------------------------------------------------ spatial filters
// Sliding box of radius r along a line of n values `stride` apart, edges repeated
// (same arithmetic on CPU and GPU).
LD_HD void slideBox(const float* in, float* out, int n, size_t stride, int r)
{
    const double norm = 1.0 / (double)(2 * r + 1);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += in[(size_t)ldClampi(i, 0, n - 1) * stride];
    for (int x = 0; x < n; ++x) {
        out[(size_t)x * stride] = (float)(sum * norm);
        sum += (double)in[(size_t)ldClampi(x + r + 1, 0, n - 1) * stride] - (double)in[(size_t)ldClampi(x - r, 0, n - 1) * stride];
    }
}

// Radius of the 3 box passes that approximate a gaussian of the given blur size.
LD_HD int boxRadiusForBlur(float size)
{
    if (size <= 0.f) return 0;
    const float sigma = 0.5f * size;
    const int r = (int)floorf(sqrtf(sigma * sigma + 0.25f));
    return r < 1 ? 1 : r;
}

// Margin (px) the spatial filters need around the output window.
LD_HD int spatialMargin(const Params& p)
{
    int m = p.medianRadius + 2 * (p.cleanRadius < 0 ? -p.cleanRadius : p.cleanRadius);
    if (p.matteBlur > 0.f) m += 3 * boxRadiusForBlur(p.matteBlur) + 1;
    return m;
}

} // namespace ldespill
