// MasterKeyerCommon.h
// Maths of MasterKeyer_OFX, shared by the CPU and CUDA back-ends (host + device).
//
// A keyer inspired by the Master Keyer of Autodesk Flame (whose algorithm is not public):
//   Degrain : the chroma (and optionally the luma) of the plate used to compute the matte is
//             smoothed, so that grain does not make the matte noisy.
//   Matte   : fraction of screen colour in each pixel (screen channel against a limit made of
//             the two other channels, relative to the same measure on the screen colour), from
//             the primary sample - or the local screen estimated from the plate. The secondary
//             sample sets the screen level that must become fully transparent (Mix).
//             Gain (and its Shadows / Midtones / Highlights corrections, weighted by the luma
//             of the pixel), clips, softness, patches, shrink / grow, blur, garbage / holdout.
//             The matte is computed on the Key In input when it is connected, else on Source.
//             Matte Mode 3D: colours are points of a 3D space (luma, Cb, Cr); the samples define
//             a tolerance volume (the segment Primary -> Secondary, Mix of it, thickened by
//             Tolerance) made transparent, surrounded by a softness envelope (3D Softness, scaled
//             by Gain and the tone corrections) where the matte goes from 0 to 1. Combined: the
//             larger of the difference matte and the 3D matte (3D fills the inside of the
//             subject, the difference matte keeps the semi-transparent edges).
//   Patches : 3 colour patches (a colour box, grown by each sample) that force colours to the
//             black or white part of the matte, or erode the edges of the matte where they are.
//   Spill   : despill relative to the screen colour, full at the edges of the key and fading
//             towards the inside over Range pixels; Hue moves the suppression towards the
//             neighbouring hues. A Spill Colour different from the Primary tints the screen
//             colour used for the despill (not the one used for the key).
//   Blend   : over Edge Size pixels inside the key, the luma of the edge can be darkened /
//             lightened and its chroma pushed towards a colour (Edge Balance) or towards the
//             blurred background.
#pragma once

#include "MKShared.h"

#if defined(__CUDACC__)
#define MK_HD __host__ __device__ inline
#else
#define MK_HD inline
#endif

namespace mk {

using mkshared::smoothstep01;
using mkshared::diClamp01;
using mkshared::diLuma;

enum ScreenSource { kScreenSamples = 0, kScreenLocal = 1 };
enum MatteMode    { kMatteDifference = 0, kMatte3D = 1, kMatteCombined = 2 };
enum PatchType    { kPatchOff = 0, kPatchBlack = 1, kPatchWhite = 2, kPatchEdge = 3 };
enum OutputView   { kOutResult = 0, kOutFrontPremult = 1, kOutSourceCC = 2, kOutMatte = 3, kOutSpill = 4, kOutEdge = 5 };

constexpr int kPatches = 3;

struct Patch {
    int   type = kPatchOff;       // kPatchOff here = patch disabled
    float lo[3] = { 0.5f, 0.5f, 0.5f };   // colour box of the samples
    float hi[3] = { 0.5f, 0.5f, 0.5f };
    float range = 0.05f;          // colour distance (from the box) fully included
    float soft = 0.05f;           // extra distance over which the patch fades
};

struct Params {
    int width = 0, height = 0;

    // matte
    float primary[3] = { 0.08f, 0.45f, 0.12f };
    float secondary[3] = { 0.06f, 0.32f, 0.09f };
    float mix = 0.f;              // 0 = primary only, 1 = secondary sets the transparent level
    int   screenSource = kScreenSamples;
    int   matteMode = kMatteDifference;
    float tolerance = 0.05f;      // 3D: radius of the tolerance volume (colour units)
    float softness3D = 0.3f;      // 3D: width of the softness envelope
    float lumaWeight = 0.5f;      // 3D: weight of the luma axis (lower: lighting changes matter less)
    float localTolerance = 0.05f; // local screen: chromaticity distance to the Primary still taken as screen
    float balance = 0.f;          // weight between the two other channels (-1 .. 1)
    float gain = 1.f;
    float shadows = 0.f, midtones = 0.f, highlights = 0.f;   // -1 .. 1: harder .. softer per tone
    float clipLow = 0.f, clipHigh = 1.f;
    float softness = 1.f;         // gamma of the matte
    int   shrink = 0;             // px, < 0 erodes, > 0 grows
    float matteBlur = 0.f;        // px
    bool  hasGarbage = false, hasHoldout = false;

    Patch patch[kPatches];

    // degrain
    bool  degrain = false;
    float grainSize = 2.f;        // px
    float grainEdges = 0.f;       // 0..1, also smooths the luma (softer edges)
    bool  grainMore = false;

    // spill
    bool  autoCC = true;
    float spillStrength = 1.f;
    float spillRange = 20.f;      // px into the key over which the despill fades
    float spillInterior = 0.2f;   // despill left deep inside the key
    float hue = 0.f;              // -1 .. 1, neighbouring hues suppressed
    float spillColour[3] = { 0.08f, 0.45f, 0.12f };   // = primary: no change

    // blend
    float edgeSize = 0.f;         // px
    float edgeLuma = 0.f;         // -1 .. 1
    float edgeBalance[3] = { 0.f, 0.f, 0.f };
    float edgeFromBG = 0.f;       // 0..1
    float bgBlur = 20.f;          // px
    bool  hasBG = false;

    int   output = kOutResult;
};

// --------------------------------------------------------------- helpers
MK_HD mkshared::Params limitParams(float bias)
{
    mkshared::Params q;
    q.math = mkshared::kMathAverage;
    q.bias = bias;
    q.limitScale = 1.f;
    return q;
}

// Fraction of screen colour S contained in c (0 = none, 1 = as much as the screen itself).
MK_HD float screenFraction(const float* c, const float* S, float balance, float* excess = nullptr)
{
    const mkshared::Params q = limitParams(balance);
    return mkshared::spillFraction(q, c, S, excess);
}

// Level of screen fraction that is fully transparent (secondary sample, Mix).
MK_HD float transparentLevel(const Params& p, const float* S)
{
    const float fs = screenFraction(p.secondary, S, p.balance);
    const float lvl = 1.f + (fs - 1.f) * diClamp01(p.mix);
    return lvl > 0.02f ? lvl : 0.02f;
}

// Weights of the shadows, midtones and highlights for a luma y (they add up to 1).
MK_HD void toneWeights(float y, float* ws, float* wm, float* wh)
{
    // wide, overlapping transitions: the correction changes progressively with the luma
    *ws = 1.f - smoothstep01(0.f, 0.6f, y);
    *wh = smoothstep01(0.2f, 1.f, y);
    const float m = 1.f - *ws - *wh;
    *wm = m > 0.f ? m : 0.f;
}

// Gain of the screen measure for a pixel: Gain x 2^(tone corrections). Higher: more transparency.
MK_HD float matteGain(const Params& p, const float* c)
{
    if (p.shadows == 0.f && p.midtones == 0.f && p.highlights == 0.f) return p.gain;
    float ws, wm, wh;
    toneWeights(diLuma(c), &ws, &wm, &wh);
    return p.gain * exp2f(p.shadows * ws + p.midtones * wm + p.highlights * wh);
}

// 3D coordinates of a colour: (luma x weight, Cb, Cr), Rec. 709.
MK_HD void to3D(const float* c, float lw, float* v)
{
    const float y = diLuma(c);
    v[0] = y * lw; v[1] = c[2] - y; v[2] = c[0] - y;
}

// 3D matte: distance of the colour to the tolerance volume (segment Primary -> Secondary x Mix,
// moved by the local screen offset S - Primary), mapped through the softness envelope.
MK_HD float matte3D(const Params& p, const float* c, const float* S, float gain)
{
    float A[3], B[3];
    for (int i = 0; i < 3; ++i) {
        const float off = S[i] - p.primary[i];
        A[i] = p.primary[i] + off;
        B[i] = p.primary[i] + (p.secondary[i] - p.primary[i]) * diClamp01(p.mix) + off;
    }
    float vc[3], va[3], vb[3];
    to3D(c, p.lumaWeight, vc); to3D(A, p.lumaWeight, va); to3D(B, p.lumaWeight, vb);
    float ab[3], ac[3], ab2 = 0.f, t = 0.f;
    for (int i = 0; i < 3; ++i) { ab[i] = vb[i] - va[i]; ac[i] = vc[i] - va[i]; ab2 += ab[i] * ab[i]; t += ab[i] * ac[i]; }
    t = ab2 > 1e-12f ? diClamp01(t / ab2) : 0.f;
    float d2 = 0.f;
    for (int i = 0; i < 3; ++i) { const float e = ac[i] - ab[i] * t; d2 += e * e; }
    const float d = sqrtf(d2) - p.tolerance;
    const float soft = p.softness3D * gain;
    if (d <= 0.f) return 0.f;
    return soft > 1e-6f ? diClamp01(d / soft) : 1.f;
}

// Raw matte (1 = subject) of a pixel from its (degrained) colour and screen colour.
MK_HD float rawMatte(const Params& p, const float* c, const float* S)
{
    const float g = matteGain(p, c);
    float a;
    if (p.matteMode == kMatte3D) {
        a = matte3D(p, c, S, g);
    } else {
        const float f = screenFraction(c, S, p.balance) * g;
        a = 1.f - diClamp01(f / transparentLevel(p, S));
        if (p.matteMode == kMatteCombined) {
            const float a3 = matte3D(p, c, S, g);
            a = a > a3 ? a : a3;
        }
    }
    const float range = p.clipHigh - p.clipLow;
    a = diClamp01((a - p.clipLow) / (range > 1e-6f ? range : 1e-6f));
    if (p.softness != 1.f && a > 0.f && a < 1.f) a = powf(a, p.softness > 1e-3f ? p.softness : 1e-3f);
    return a;
}

// Colour patches.
MK_HD float applyPatches(const Params& p, const float* c, float a)
{
    for (int k = 0; k < kPatches; ++k) {
        const Patch& pt = p.patch[k];
        if (pt.type == kPatchOff) continue;
        float d2 = 0.f;                             // squared distance to the colour box
        for (int i = 0; i < 3; ++i) {
            const float e = c[i] < pt.lo[i] ? pt.lo[i] - c[i] : (c[i] > pt.hi[i] ? c[i] - pt.hi[i] : 0.f);
            d2 += e * e;
        }
        const float d = sqrtf(d2);
        const float w = 1.f - smoothstep01(pt.range, pt.range + pt.soft, d);
        if (w <= 0.f) continue;
        if (pt.type == kPatchBlack)      a *= 1.f - w;
        else if (pt.type == kPatchWhite) a = a > w ? a : w;
        else                             a *= 1.f - w * (1.f - a);   // erodes semi-transparent edges only
    }
    return a;
}

// Pass 1 (local screen only): screen seeds (premultiplied colour, weight). A pixel is taken as
// screen when its chromaticity (colour / sum) is within localTolerance of the Primary's and it
// holds at least a quarter of the Primary's screen measure: darker or brighter parts of an uneven
// screen are seeds too, while edges (mixed with the subject colour) are not. The seeds carry the
// colour used for the matte (degrained or not), so that the screen matches it exactly.
MK_HD void screenSeed(const Params& p, const float* matteColour, float* base4)
{
    const float* c = matteColour;
    const float sc = c[0] + c[1] + c[2], sp = p.primary[0] + p.primary[1] + p.primary[2];
    float w = 0.f;
    if (sc > 1e-6f && sp > 1e-6f) {
        float d = 0.f;
        for (int k = 0; k < 3; ++k) { const float e = fabsf(c[k] / sc - p.primary[k] / sp); d = e > d ? e : d; }
        const float f = screenFraction(c, p.primary, p.balance);
        w = d <= p.localTolerance && f >= 0.25f ? 1.f : 0.f;
    }
    base4[0] = c[0] * w; base4[1] = c[1] * w; base4[2] = c[2] * w; base4[3] = w;
}

// Screen colour of a pixel: primary sample, or pushed local screen when available.
MK_HD void screenAt(const Params& p, const float* pushed4, float* S)
{
    if (p.screenSource == kScreenLocal && pushed4 && pushed4[3] > 1e-12f) {
        const float k = 1.f / pushed4[3];
        S[0] = pushed4[0] * k; S[1] = pushed4[1] * k; S[2] = pushed4[2] * k;
        return;
    }
    S[0] = p.primary[0]; S[1] = p.primary[1]; S[2] = p.primary[2];
}

// Matte of a pixel before the spatial filters: raw matte + patches.
MK_HD float mattePixel(const Params& p, const float* matteColour, const float* S)
{
    return applyPatches(p, matteColour, rawMatte(p, matteColour, S));
}

// Garbage (forces transparent) and holdout (forces opaque).
MK_HD float garbageHoldout(const Params& p, float a, const float* garbage, const float* holdout)
{
    if (p.hasGarbage && garbage) a *= 1.f - diClamp01(*garbage);
    if (p.hasHoldout && holdout) { const float h = diClamp01(*holdout); a = a > h ? a : h; }
    return a;
}

// ------------------------------------------------------------------ degrain
// Rec. 709 colour differences and luma, stored as (Cb, Cr, Y, 0): with Edges = 0 only the
// first two channels need blurring.
MK_HD void toYCC(const float* c, float* o)
{
    const float y = diLuma(c);
    o[0] = c[2] - y; o[1] = c[0] - y; o[2] = y; o[3] = 0.f;
}
MK_HD void fromYCC(const float* ycc, const float* yccBlur, float edges, float* c)
{
    const float y = ycc[2] + (yccBlur[2] - ycc[2]) * edges;
    const float cb = yccBlur[0], cr = yccBlur[1];
    c[0] = cr + y;
    c[2] = cb + y;
    c[1] = (y - 0.2126f * c[0] - 0.0722f * c[2]) / 0.7152f;
}
// Channels of the YCC buffer to blur.
MK_HD int degrainChannels(const Params& p) { return p.grainEdges > 0.f ? 3 : 2; }
MK_HD float grainBlurSize(const Params& p) { return p.grainSize * (p.grainMore ? 2.f : 1.f); }

// --------------------------------------------------------------- final pixel
//   src4 : plate RGBA          S : screen colour          a : final matte
//   dist : distance (px) to the nearest screen pixel     bg3 : background or null
//   bgBlur3 : blurred background or null
MK_HD void finalPixel(const Params& p, const float* src4, const float* S, float a, float dist,
                      const float* bg3, const float* bgBlur3, float* out4)
{
    float c[3] = { src4[0], src4[1], src4[2] };

    // spill (Auto CC): screen channel limited by the two others, relative to the screen colour
    float spill = 0.f;
    if (p.autoCC && p.spillStrength > 0.f) {
        const float t = smoothstep01(0.f, p.spillRange > 1e-3f ? p.spillRange : 1e-3f, dist);
        const float st = p.spillStrength * (1.f + (p.spillInterior - 1.f) * t);
        // screen colour of the spill: S tinted by Spill Colour / Primary (identical when equal)
        float Sp[3] = { S[0], S[1], S[2] };
        const bool tinted = p.spillColour[0] != p.primary[0] || p.spillColour[1] != p.primary[1] ||
                            p.spillColour[2] != p.primary[2];
        if (tinted)
            for (int i = 0; i < 3; ++i)
                Sp[i] = p.primary[i] > 1e-6f ? S[i] * p.spillColour[i] / p.primary[i] : p.spillColour[i];
        int s, o1, o2;
        mkshared::screenChannels(Sp, &s, &o1, &o2);
        // a spill colour richer in one of the two other channels than the screen moves the
        // limit towards the other one (that channel is contaminated too)
        float bias = p.hue;
        const float d1 = Sp[o1] + Sp[o2], d0 = S[o1] + S[o2];
        if (tinted && d1 > 1e-6f && d0 > 1e-6f) bias -= 2.f * ((Sp[o2] - Sp[o1]) / d1 - (S[o2] - S[o1]) / d0);
        bias = bias < -1.f ? -1.f : (bias > 1.f ? 1.f : bias);
        const mkshared::Params q = limitParams(bias);
        const float lim = mkshared::channelLimit(q, c[o1], c[o2]);
        const float ex = c[s] - lim;
        if (ex > 0.f) { spill = ex * st; c[s] -= spill; }
    }
    if (p.output == kOutSourceCC) {         // despilled plate, before the blend
        out4[0] = c[0]; out4[1] = c[1]; out4[2] = c[2]; out4[3] = a;
        return;
    }

    // blend: edge region inside the key
    float wE = 0.f;
    if (p.edgeSize > 0.f && a > 0.f) {
        wE = 1.f - smoothstep01(0.f, p.edgeSize, dist);
        if (wE > 0.f) {
            const float k = 1.f + p.edgeLuma * wE;
            c[0] *= k; c[1] *= k; c[2] *= k;
            const float Lb = diLuma(p.edgeBalance);
            for (int i = 0; i < 3; ++i) c[i] += wE * (p.edgeBalance[i] - Lb);
            if (bgBlur3 && p.hasBG && p.edgeFromBG > 0.f) {
                const float Lc = diLuma(c), Lg = diLuma(bgBlur3);
                const float w = wE * p.edgeFromBG;
                for (int i = 0; i < 3; ++i) c[i] += w * ((bgBlur3[i] - Lg) - (c[i] - Lc));
            }
        }
    }

    switch (p.output) {
    case kOutFrontPremult: out4[0] = c[0] * a; out4[1] = c[1] * a; out4[2] = c[2] * a; break;
    case kOutMatte:        out4[0] = out4[1] = out4[2] = a; break;
    case kOutSpill:        out4[0] = out4[1] = out4[2] = spill; break;
    case kOutEdge:         out4[0] = out4[1] = out4[2] = wE; break;
    default:
        if (bg3 && p.hasBG) {           // composite over the (opaque) background
            for (int i = 0; i < 3; ++i) out4[i] = c[i] * a + bg3[i] * (1.f - a);
            out4[3] = 1.f;
            return;
        }
        for (int i = 0; i < 3; ++i) out4[i] = c[i] * a;
        break;
    }
    out4[3] = a;
}

} // namespace mk
