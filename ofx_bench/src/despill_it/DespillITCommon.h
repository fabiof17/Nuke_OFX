// DespillITCommon.h
// Maths of despill_IT_OFX shared by the CPU and CUDA back-ends (host + device), so that both
// paths give the same image.
//
// Spatially aware despill:
//   1. key        : alpha of a keyer (Key input) or an internal key computed from the spill.
//   2. screen     : global screen colour, or the local screen estimated from the screen areas
//                   of the plate (push-pull), or a clean plate.
//   3. spill      : fraction f of the local screen colour contained in the pixel (limit of the
//                   screen channel by the two others, relative to the same limit on the screen).
//   4. strength   : full near the screen, fading towards the inside of the subject (exact
//                   distance to the screen), reduced on protected hues / Protect input.
//   5. despill    : limits the screen channel (classic), removes f x the chroma of the local
//                   screen (luminance kept), or removes f x the local screen light (unmix).
//   6. edge colour: the chroma of semi-transparent edges is rebuilt from the opaque core colour
//                   propagated outwards (push-pull), keeping the pixel luminance.
//   7. spill light: the removed screen light can be replaced by the blurred background.
//   8. light wrap : blurred background added on the inner edges of the subject.
#pragma once

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#define DI_HD __host__ __device__ inline
#else
#define DI_HD inline
#endif

namespace dspit {

enum ScreenSource { kScreenGlobal = 0, kScreenFromPlate = 1, kScreenCleanPlate = 2 };
enum KeySource    { kKeyInternal = 0, kKeyInput = 1 };
enum DespillMath  { kMathAverage = 0, kMathMax = 1 };
enum DespillMode  { kModeChannel = 0, kModeNeutralize = 1, kModeSubtract = 2 };
enum SpillLight   { kLightNone = 0, kLightBackground = 1 };
enum OutputView   { kOutResult = 0, kOutSpillMap = 1, kOutStrength = 2, kOutKey = 3, kOutScreen = 4, kOutCore = 5 };
enum AlphaOut     { kAlphaSource = 0, kAlphaKey = 1, kAlphaSpill = 2 };

struct Params {
    int width = 0, height = 0;

    // screen
    float screen[3] = { 0.08f, 0.45f, 0.12f };   // global screen colour (linear)
    int   screenSource = kScreenFromPlate;
    // detection
    int   math = kMathAverage;
    float bias = 0.f;
    float limitScale = 1.f;
    // key
    int   keySource = kKeyInternal;
    float keyLow = 0.4f, keyHigh = 0.9f;          // internal key: spill fraction -> alpha 1 .. 0
    float screenBelow = 0.05f;                    // alpha <= : screen pixel
    float coreAbove = 0.95f;                      // alpha >= : opaque core pixel
    // strength
    float strength = 1.f;
    bool  spatial = true;
    float edgeWidth = 4.f;                        // px at full strength from the screen
    float edgeFalloff = 40.f;                     // px of fade towards the inside
    float interior = 0.15f;                       // strength far from the screen
    // protection
    float protectHue = 30.f;                      // degrees
    float protectRange = 0.f;                     // degrees (0 = off)
    float protectSoftness = 15.f;                 // degrees
    float protectAmount = 1.f;
    bool  hasProtect = false;
    // despill
    int   mode = kModeChannel;
    float edgeRestore = 0.75f;
    int   spillLight = kLightNone;
    float bgGain = 1.f;
    float bgBlur = 20.f;                          // px
    float wrapAmount = 0.f;
    bool  hasBG = false;
    // output
    int   output = kOutResult;
    int   alphaOut = kAlphaSource;
    float mix = 1.f;
    bool  showZone = false;                       // Show Overlay: edge zone tinted blue (viewing aid)
};

// ------------------------------------------------------------------ helpers
DI_HD float diClamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
DI_HD int   diClampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
DI_HD float diLuma(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }
DI_HD float smoothstep01(float e0, float e1, float x)
{
    if (e1 <= e0) return x < e0 ? 0.f : 1.f;
    const float t = diClamp01((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}

// Screen channel (largest component of the screen colour) and the two others.
DI_HD void screenChannels(const float* S, int* s, int* o1, int* o2)
{
    if (S[2] > S[1] && S[2] >= S[0])      { *s = 2; *o1 = 0; *o2 = 1; }   // blue
    else if (S[0] > S[1] && S[0] > S[2])  { *s = 0; *o1 = 1; *o2 = 2; }   // red
    else                                  { *s = 1; *o1 = 0; *o2 = 2; }   // green
}

// Limit of the screen channel from the two others (Average or Max, with bias).
DI_HD float channelLimit(const Params& p, float v1, float v2)
{
    const float b = p.bias < -1.f ? -1.f : (p.bias > 1.f ? 1.f : p.bias);
    float lim;
    if (p.math == kMathMax) {
        const float w1 = 1.f - b < 1.f ? 1.f - b : 1.f, w2 = 1.f + b < 1.f ? 1.f + b : 1.f;
        const float a1 = w1 * v1, a2 = w2 * v2;
        lim = a1 > a2 ? a1 : a2;
    } else {
        lim = 0.5f * ((1.f - b) * v1 + (1.f + b) * v2);
    }
    return lim * p.limitScale;
}

// Spill of pixel c relative to screen S: a = excess of the screen channel over its limit,
// f = a / (the same excess for the screen colour), i.e. the fraction of the screen in c.
DI_HD float spillFraction(const Params& p, const float* c, const float* S, float* excess)
{
    int s, o1, o2;
    screenChannels(S, &s, &o1, &o2);
    const float a = c[s] - channelLimit(p, c[o1], c[o2]);
    const float aS = S[s] - channelLimit(p, S[o1], S[o2]);
    const float ap = a > 0.f ? a : 0.f;
    if (excess) *excess = ap;
    return aS > 1e-6f ? ap / aS : 0.f;
}

// Internal key: alpha 1 (no spill) .. 0 (as much screen as the screen itself).
DI_HD float internalKey(const Params& p, float f) { return 1.f - smoothstep01(p.keyLow, p.keyHigh, f); }

// Hue in degrees [0, 360) and saturation (HSV) of a colour.
DI_HD void hueSat(const float* c, float* hue, float* sat)
{
    const float mx = c[0] > c[1] ? (c[0] > c[2] ? c[0] : c[2]) : (c[1] > c[2] ? c[1] : c[2]);
    const float mn = c[0] < c[1] ? (c[0] < c[2] ? c[0] : c[2]) : (c[1] < c[2] ? c[1] : c[2]);
    const float d = mx - mn;
    *sat = mx > 1e-6f ? d / mx : 0.f;
    if (d <= 1e-8f) { *hue = 0.f; return; }
    float h;
    if (mx == c[0])      h = (c[1] - c[2]) / d;
    else if (mx == c[1]) h = 2.f + (c[2] - c[0]) / d;
    else                 h = 4.f + (c[0] - c[1]) / d;
    h *= 60.f;
    if (h < 0.f) h += 360.f;
    *hue = h;
}

// Protection of hues around protectHue (skin, eyes...), weighted by saturation.
DI_HD float hueProtection(const Params& p, const float* c)
{
    if (p.protectRange <= 0.f && p.protectSoftness <= 0.f) return 0.f;
    float h, s;
    hueSat(c, &h, &s);
    float dh = fabsf(h - p.protectHue);
    if (dh > 180.f) dh = 360.f - dh;
    const float inRange = 1.f - smoothstep01(p.protectRange, p.protectRange + p.protectSoftness, dh);
    return inRange * smoothstep01(0.03f, 0.12f, s);
}

// Despill strength from the distance (px) to the nearest screen pixel.
DI_HD float spatialStrength(const Params& p, float dist)
{
    if (!p.spatial) return 1.f;
    const float t = smoothstep01(p.edgeWidth, p.edgeWidth + p.edgeFalloff, dist);
    return 1.f + (p.interior - 1.f) * t;
}

// ----------------------------------------------------------- per-pixel stages
// Pass 1: key with the global screen colour, and the screen seed of the local screen
// estimation (premultiplied colour, weight).
DI_HD float keyPass1(const Params& p, const float* c, const float* keyIn, float* screenBase4)
{
    float alpha;
    if (p.keySource == kKeyInput && keyIn) alpha = diClamp01(*keyIn);
    else alpha = internalKey(p, spillFraction(p, c, p.screen, nullptr));
    const float w = alpha <= p.screenBelow ? 1.f : 0.f;
    screenBase4[0] = c[0] * w; screenBase4[1] = c[1] * w; screenBase4[2] = c[2] * w; screenBase4[3] = w;
    return alpha;
}

// Local screen colour at a pixel: pushed screen colours, clean plate or global colour.
DI_HD void localScreen(const Params& p, const float* pushed4, const float* clean3, float* S)
{
    if (p.screenSource == kScreenCleanPlate && clean3) { S[0] = clean3[0]; S[1] = clean3[1]; S[2] = clean3[2]; return; }
    if (p.screenSource == kScreenFromPlate && pushed4 && pushed4[3] > 1e-12f) {
        const float k = 1.f / pushed4[3];
        S[0] = pushed4[0] * k; S[1] = pushed4[1] * k; S[2] = pushed4[2] * k;
        return;
    }
    S[0] = p.screen[0]; S[1] = p.screen[1]; S[2] = p.screen[2];
}

// Despill of a colour with the local screen; returns f (spill fraction), strength in *st.
DI_HD float despillColor(const Params& p, const float* c, const float* S, float strength, float* out)
{
    float excess;
    const float f = spillFraction(p, c, S, &excess);
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
    if (f <= 0.f || strength <= 0.f) return f;
    if (p.mode == kModeChannel) {
        int s, o1, o2;
        screenChannels(S, &s, &o1, &o2);
        out[s] -= excess * strength;
    } else {
        const float L = p.mode == kModeNeutralize ? diLuma(S) : 0.f, fe = f * strength;
        out[0] -= fe * (S[0] - L); out[1] -= fe * (S[1] - L); out[2] -= fe * (S[2] - L);
    }
    return f;
}

// Pass 2: final key, despill strength (protection only, spatial applied later), core seed.
// Returns the key; writes the screen mask (1 = screen) for the distance transform.
DI_HD float keyPass2(const Params& p, const float* c, const float* keyIn, const float* S, float protect,
                     float* coreBase4, float* screenMask)
{
    float alpha;
    if (p.keySource == kKeyInput && keyIn) alpha = diClamp01(*keyIn);
    else alpha = internalKey(p, spillFraction(p, c, S, nullptr));
    *screenMask = alpha <= p.screenBelow ? 1.f : 0.f;
    // core colour: despilled with the protection only (the core is far from the screen)
    float d[3];
    despillColor(p, c, S, p.strength * (1.f - protect), d);
    const float w = alpha >= p.coreAbove ? 1.f : 0.f;
    coreBase4[0] = d[0] * w; coreBase4[1] = d[1] * w; coreBase4[2] = d[2] * w; coreBase4[3] = w;
    return alpha;
}

// Protection of a pixel: hue protection and Protect input, times the amount.
DI_HD float protection(const Params& p, const float* c, const float* protectIn)
{
    float pr = hueProtection(p, c);
    if (p.hasProtect && protectIn) { const float m = diClamp01(*protectIn); pr = pr > m ? pr : m; }
    return diClamp01(pr * p.protectAmount);
}

// Final colour of a pixel.
//   src4: plate RGBA, S: local screen, alpha: key, dist: distance to the screen (px),
//   core4: pushed core colour (premultiplied, weight) or null, bg3: blurred BG or null,
//   wrapEdge: blurred (1 - key) for the light wrap, protectIn / mask: optional mattes.
DI_HD void finalPixel(const Params& p, const float* src4, const float* S, float alpha, float dist,
                      const float* core4, const float* bg3, float wrapEdge, const float* protectIn,
                      float mask, float* out4)
{
    const float pr = protection(p, src4, protectIn);
    const float st = p.strength * spatialStrength(p, dist) * (1.f - pr);
    float c[3];
    const float f = despillColor(p, src4, S, st, c);
    const float fe = f * st;

    // edge colour from the opaque core, luminance kept
    if (core4 && p.edgeRestore > 0.f && core4[3] > 1e-12f && fe > 0.f) {
        const float k = 1.f / core4[3];
        const float C[3] = { core4[0] * k, core4[1] * k, core4[2] * k };
        const float LC = diLuma(C), Lc = diLuma(c);
        if (LC > 1e-6f) {
            const float w = p.edgeRestore * (1.f - alpha) * (fe < 1.f ? fe : 1.f);
            const float r = Lc / LC;
            for (int i = 0; i < 3; ++i) c[i] += (C[i] * r - c[i]) * w;
        }
    }

    // removed screen light replaced by the background light (Neutralize: in place of the
    // neutral light it left; Subtract: exactly the composite of the edge over the background)
    if (fe > 0.f && p.spillLight == kLightBackground && bg3 && p.hasBG) {
        const float LS = p.mode == kModeNeutralize ? diLuma(S) : 0.f;
        for (int i = 0; i < 3; ++i) c[i] += fe * (p.bgGain * bg3[i] - LS);
    }

    // light wrap on the inner edges
    if (p.wrapAmount > 0.f && bg3 && p.hasBG) {
        const float w = p.wrapAmount * alpha * wrapEdge;
        for (int i = 0; i < 3; ++i) c[i] += w * bg3[i];
    }

    float o[4] = { c[0], c[1], c[2], src4[3] };
    switch (p.output) {
    case kOutSpillMap:  o[0] = o[1] = o[2] = fe; break;
    case kOutStrength:  o[0] = o[1] = o[2] = st; break;
    case kOutKey:       o[0] = o[1] = o[2] = alpha; break;
    case kOutScreen:    o[0] = S[0]; o[1] = S[1]; o[2] = S[2]; break;
    case kOutCore:
        if (core4 && core4[3] > 1e-12f) { const float k = 1.f / core4[3]; o[0] = core4[0] * k; o[1] = core4[1] * k; o[2] = core4[2] * k; }
        else o[0] = o[1] = o[2] = 0.f;
        break;
    default: break;
    }
    if (p.alphaOut == kAlphaKey) o[3] = alpha;
    else if (p.alphaOut == kAlphaSpill) o[3] = diClamp01(fe);

    const float k = p.mix * mask;
    for (int i = 0; i < 4; ++i) out4[i] = src4[i] + (o[i] - src4[i]) * k;

    // Show Overlay: the subject pixels from the screen up to Edge Width + Edge Falloff are tinted
    // blue, 50 % over Edge Width, fading to 20 % at the end of Edge Falloff (RGB only, alpha kept)
    if (p.showZone && p.spatial && dist > 0.f && dist <= p.edgeWidth + p.edgeFalloff) {
        const float a = 0.5f - 0.3f * smoothstep01(p.edgeWidth, p.edgeWidth + p.edgeFalloff, dist);
        const float B[3] = { 0.1f, 0.35f, 1.f };
        for (int i = 0; i < 3; ++i) out4[i] += (B[i] - out4[i]) * a;
    }
}

// ------------------------------------------------------------- push-pull (RGBW)
DI_HD void pullPixel(const float* fine, int fw, int fh, int cx, int cy, float* out4)
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

DI_HD void bilinearClamp4(const float* img, int w, int h, float x, float y, float* o)
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

DI_HD void pushPixel(float* fine4, const float* coarse, int cw, int ch, int fx, int fy)
{
    float c[4];
    bilinearClamp4(coarse, cw, ch, ((float)fx + 0.5f) * 0.5f - 0.5f, ((float)fy + 0.5f) * 0.5f - 0.5f, c);
    const float k = 1.f - diClamp01(fine4[3]);
    fine4[0] += k * c[0]; fine4[1] += k * c[1]; fine4[2] += k * c[2]; fine4[3] += k * c[3];
}

// --------------------------------------------- exact distance transform (to the screen)
struct Seed { int x, y; };

DI_HD void edtColumn(const float* mask, int w, int h, int x, int* nearestY)
{
    int last = -1;
    for (int y = 0; y < h; ++y) {
        if (mask[(size_t)y * w + x] > 0.5f) last = y;
        nearestY[(size_t)y * w + x] = last;
    }
    int next = -1;
    for (int y = h - 1; y >= 0; --y) {
        if (mask[(size_t)y * w + x] > 0.5f) next = y;
        const int up = nearestY[(size_t)y * w + x];
        int best = up;
        if (up < 0) best = next;
        else if (next >= 0 && next - y < y - up) best = next;
        nearestY[(size_t)y * w + x] = best;
    }
}

// Lower envelope of the parabolas, exact integer arithmetic (see push_IT_OFX).
// Element i of the row lives at ny[i * s], v[i * s], zn[i * s], zd[i * s], out[i * s].
DI_HD void edtRow(const int* ny, int w, int y, size_t s, int* v, long long* zn, int* zd, Seed* out)
{
    int k = -1;
    for (int q = 0; q < w; ++q) {
        const int nq = ny[(size_t)q * s];
        if (nq < 0) continue;
        const long long dq = y - nq;
        const long long fq = dq * dq + (long long)q * q;
        while (true) {
            if (k < 0) { k = 0; v[0] = q; break; }
            const int pp = v[(size_t)k * s];
            const long long dp = y - ny[(size_t)pp * s];
            const long long num = fq - (dp * dp + (long long)pp * pp);
            const long long den = 2LL * (q - pp);
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
        while (j < k && zn[(size_t)(j + 1) * s] < (long long)x * zd[(size_t)(j + 1) * s]) ++j;
        const int vj = v[(size_t)j * s];
        out[(size_t)x * s].x = vj; out[(size_t)x * s].y = ny[(size_t)vj * s];
    }
}

DI_HD float seedDistance(Seed sd, int x, int y)
{
    if (sd.x < 0) return 1e30f;
    const long long dx = sd.x - x, dy = sd.y - y;
    return sqrtf((float)(dx * dx + dy * dy));
}

// ------------------------------------------------------------- sliding box blur
DI_HD void slideBox(const float* in, float* out, int n, size_t stride, int nch, int r)
{
    const double norm = 1.0 / (double)(2 * r + 1);
    double sum[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = -r; i <= r; ++i) {
        const float* q = in + (size_t)diClampi(i, 0, n - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += q[c];
    }
    for (int x = 0; x < n; ++x) {
        float* o = out + (size_t)x * stride;
        for (int c = 0; c < nch; ++c) o[c] = (float)(sum[c] * norm);
        const float* qa = in + (size_t)diClampi(x + r + 1, 0, n - 1) * stride;
        const float* qs = in + (size_t)diClampi(x - r, 0, n - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += (double)qa[c] - (double)qs[c];
    }
}

// Background blur on a reduced image: large blurs are computed on the BG reduced by an
// integer factor (area average), then scaled back up bilinearly (invisible on a blurred
// background, and several times faster).
// Picks the reduction factor f and the box radius r on the reduced image: the largest f
// (<= blur / 8) for which an integer radius reproduces the full-resolution blur within 10 %
// of its variance, once the softening of the reduction and of the bilinear scale-up is
// taken into account. Small blurs are not reduced (f = 1).
DI_HD void bgReducePlan(float blur, int* f, int* r);

// Area average of the f x f block (X, Y) of a w x h image with nch floats per pixel.
DI_HD void downsamplePixel(const float* in, int w, int h, int nch, int f, int X, int Y, float* out)
{
    const int x0 = X * f, y0 = Y * f;
    const int x1 = x0 + f < w ? x0 + f : w, y1 = y0 + f < h ? y0 + f : h;
    float s[3] = { 0.f, 0.f, 0.f };
    for (int y = y0; y < y1; ++y) {
        const float* q = in + (size_t)nch * ((size_t)y * w + x0);
        for (int x = x0; x < x1; ++x, q += nch)
            for (int c = 0; c < nch; ++c) s[c] += q[c];
    }
    const float k = 1.f / (float)((x1 - x0) * (y1 - y0));
    for (int c = 0; c < nch; ++c) out[c] = s[c] * k;
}

// Bilinear sample of the reduced image (ws x hs, factor f) at full-resolution pixel (x, y).
DI_HD void sampleReduced(const float* img, int ws, int hs, int nch, int f, int x, int y, float* out)
{
    float u = ((float)x + 0.5f) / (float)f - 0.5f;
    float v = ((float)y + 0.5f) / (float)f - 0.5f;
    u = u < 0.f ? 0.f : (u > (float)(ws - 1) ? (float)(ws - 1) : u);
    v = v < 0.f ? 0.f : (v > (float)(hs - 1) ? (float)(hs - 1) : v);
    const int u0 = (int)u, v0 = (int)v;
    const int u1 = u0 + 1 < ws ? u0 + 1 : u0, v1 = v0 + 1 < hs ? v0 + 1 : v0;
    const float fu = u - (float)u0, fv = v - (float)v0;
    for (int c = 0; c < nch; ++c) {
        const float a = img[(size_t)nch * ((size_t)v0 * ws + u0) + c], b = img[(size_t)nch * ((size_t)v0 * ws + u1) + c];
        const float d = img[(size_t)nch * ((size_t)v1 * ws + u0) + c], e = img[(size_t)nch * ((size_t)v1 * ws + u1) + c];
        const float top = a + (b - a) * fu, bot = d + (e - d) * fu;
        out[c] = top + (bot - top) * fv;
    }
}

DI_HD int boxRadiusForBlur(float size)
{
    if (size <= 0.f) return 0;
    const float sigma = 0.5f * size;
    const int r = (int)floorf(sqrtf(sigma * sigma + 0.25f));
    return r < 1 ? 1 : r;
}

DI_HD void bgReducePlan(float blur, int* f, int* r)
{
    const int rf = boxRadiusForBlur(blur);
    *f = 1; *r = rf;
    if (rf <= 0) return;
    const float V = (float)rf * (float)(rf + 1);                 // variance of the 3 full-size boxes
    for (int c = (int)floorf(blur / 8.f); c >= 2; --c) {
        const float c2 = (float)(c * c);
        const float Vs = (V - (c2 - 1.f) / 12.f - c2 / 6.f) / c2;   // variance left for the reduced blur
        if (Vs < 6.f) continue;
        int rs = (int)floorf(sqrtf(Vs + 0.25f) - 0.5f);
        float e = fabsf((float)(rs * (rs + 1)) - Vs) / Vs;
        const float e2 = fabsf((float)((rs + 1) * (rs + 2)) - Vs) / Vs;
        if (e2 < e) { ++rs; e = e2; }
        if (e < 0.10f) { *f = c; *r = rs; return; }
    }
}

} // namespace dspit
