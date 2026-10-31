/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX, a GPU-accelerated derivative of the
 * openfx-misc CornerPin plugin <https://github.com/NatronGitHub/openfx-misc>.
 * (C) 2018-2021 The Natron Developers
 * (C) 2013-2018 INRIA
 * (C) 2026 CornerPinEx_OFX contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * Shared sampling code for CornerPinEx_OFX.
 *
 * Everything in this header compiles both as plain C++ (CPU fallback) and as
 * CUDA device code, so that the CPU and GPU paths produce the same image.
 *
 * Conventions (same as the OpenFX / Natron code):
 * - coordinates are in pixels, the center of pixel (i,j) is at (i+0.5, j+0.5);
 * - values are in the native range of the image (0..255, 0..65535 or float);
 * - inverse transforms map a destination pixel center to a source position.
 */

#ifndef CORNERPINEX_OFX_SAMPLER_H
#define CORNERPINEX_OFX_SAMPLER_H

#include <math.h>
#include <stddef.h>

#if defined(__CUDACC__)
#  define CPG_HD __host__ __device__ __forceinline__
#else
#  define CPG_HD inline
#endif

namespace cpg {

// The first 10 values match OFX::FilterEnum (ofxsFilter.h), the others are
// the high-quality filters added by CornerPinEx_OFX.
enum FilterType
{
    eImpulse = 0,
    eBox,
    eBilinear,
    eCubic,
    eKeys,
    eSimon,
    eRifman,
    eMitchell,
    eParzen,
    eNotch,
    eEWAGaussian,   // elliptical weighted average, Gaussian kernel (HQ, smooth)
    eEWALanczos3,   // elliptical (anisotropic) Lanczos-3 kernel (HQ, sharp)
    eFilterCount
};

CPG_HD bool isEWA(int f) { return f == eEWAGaussian || f == eEWALanczos3; }

#define CPG_MAX_LEVELS 16
#define CPG_EWA_MAX_AXIS 12.f       // max ellipse semi-axis (in texels of the chosen mip level)
#define CPG_BOX_MAX_TEXELS 64       // max texels per axis for the box filter

// One level of a float image pyramid. Level 0 is the source image itself.
struct FImage
{
    const float* data; // nComp floats per pixel, row-major
    int width;
    int height;
    long long stride; // in floats
};

struct Pyramid
{
    FImage level[CPG_MAX_LEVELS];
    int nLevels;
    int x1, y1;       // absolute pixel coordinates of the bottom-left pixel of level 0
};

// 3x3 matrix, row-major, float (enough precision for pixel coordinates)
struct Mat3
{
    float m[9];
};

struct SampleParams
{
    int filter;
    bool clamp;
    bool blackOutside;
};

CPG_HD int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
CPG_HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
CPG_HD float minf(float a, float b) { return a < b ? a : b; }
CPG_HD float maxf(float a, float b) { return a > b ? a : b; }

// Fetch texel (i,j) (level-local indices). Returns false when the texel is outside and black.
template <int NC>
CPG_HD bool fetch(const FImage& im, int i, int j, bool blackOutside, float* out)
{
    if ( (i < 0) || (j < 0) || (i >= im.width) || (j >= im.height) ) {
        if (blackOutside || (im.width <= 0) || (im.height <= 0)) {
            for (int c = 0; c < NC; ++c) {
                out[c] = 0.f;
            }
            return false;
        }
        i = clampi(i, 0, im.width - 1);
        j = clampi(j, 0, im.height - 1);
    }
    const float* p = im.data + (long long)j * im.stride + (long long)i * NC;
    for (int c = 0; c < NC; ++c) {
        out[c] = p[c];
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// 1D interpolators: identical formulas to ofxsFilter.h (Natron)

CPG_HD float clampVal(float I, float Ic, float In)
{
    float lo = minf(Ic, In), hi = maxf(Ic, In);
    return I < lo ? lo : (I > hi ? hi : I);
}

CPG_HD float interp1(int filter, float Ip, float Ic, float In, float Ia, float d, bool clamp)
{
    float I;
    switch (filter) {
    case eBilinear:
        return Ic + d * (In - Ic);
    case eCubic:
        I = Ic + d * d * ( (-3 * Ic + 3 * In ) + d * (2 * Ic - 2 * In ) );
        break;
    case eKeys:
        I = Ic  + d * ( (-Ip + In ) + d * ( (2 * Ip - 5 * Ic + 4 * In - Ia ) + d * (-Ip + 3 * Ic - 3 * In + Ia ) ) ) / 2;
        break;
    case eSimon:
        I = Ic  + d * ( (-3 * Ip + 3 * In ) + d * ( (6 * Ip - 9 * Ic + 6 * In - 3 * Ia ) + d * (-3 * Ip + 5 * Ic - 5 * In + 3 * Ia ) ) ) / 4;
        break;
    case eRifman:
        I = Ic  + d * ( (-Ip + In ) + d * ( (2 * Ip - 2 * Ic + In - Ia ) + d * (-Ip + Ic - In + Ia ) ) );
        break;
    case eMitchell:
        I = ( Ip + 16 * Ic + In + d * ( (-9 * Ip + 9 * In ) + d * ( (15 * Ip - 36 * Ic + 27 * In - 6 * Ia ) + d * (-7 * Ip + 21 * Ic - 21 * In + 7 * Ia ) ) ) ) / 18;
        break;
    case eParzen:
        return ( Ip + 4 * Ic + In + d * ( (-3 * Ip + 3 * In ) + d * ( (3 * Ip - 6 * Ic + 3 * In ) + d * (-Ip + 3 * Ic - 3 * In + Ia ) ) ) ) / 6;
    case eNotch:
        return ( Ip + 2 * Ic + In + d * ( (-2 * Ip + 2 * In ) + d * ( (Ip - Ic - In + Ia ) ) ) ) / 4;
    default:
        return Ic;
    }
    if (clamp) {
        I = clampVal(I, Ic, In);
    }
    return I;
}

// Point interpolation at level-0 position (fx,fy) (absolute pixel coordinates),
// with the classic filters (everything except box and EWA).
// Returns false when every texel used was outside and black.
template <int NC>
CPG_HD bool interpolate(const Pyramid& pyr, float fx, float fy, const SampleParams& sp, float* out)
{
    const FImage& im = pyr.level[0];
    // local coordinates
    float lx = fx - pyr.x1;
    float ly = fy - pyr.y1;

    if ( (sp.filter == eImpulse) || (sp.filter == eBox) ) {
        int mx = (int)floorf(lx);
        int my = (int)floorf(ly);
        return fetch<NC>(im, mx, my, sp.blackOutside, out);
    }

    int cx = (int)floorf(lx - 0.5f);
    int cy = (int)floorf(ly - 0.5f);
    float dx = clampf(lx - 0.5f - cx, 0.f, 1.f);
    float dy = clampf(ly - 0.5f - cy, 0.f, 1.f);
    bool any = false;

    if ( (sp.filter == eBilinear) || (sp.filter == eCubic) ) {
        float Icc[NC], Inc[NC], Icn[NC], Inn[NC];
        any |= fetch<NC>(im, cx,     cy,     sp.blackOutside, Icc);
        any |= fetch<NC>(im, cx + 1, cy,     sp.blackOutside, Inc);
        any |= fetch<NC>(im, cx,     cy + 1, sp.blackOutside, Icn);
        any |= fetch<NC>(im, cx + 1, cy + 1, sp.blackOutside, Inn);
        for (int c = 0; c < NC; ++c) {
            float Ic = interp1(sp.filter, 0.f, Icc[c], Inc[c], 0.f, dx, sp.clamp);
            float In = interp1(sp.filter, 0.f, Icn[c], Inn[c], 0.f, dx, sp.clamp);
            out[c] = interp1(sp.filter, 0.f, Ic, In, 0.f, dy, sp.clamp);
        }
        if (!any) {
            for (int c = 0; c < NC; ++c) {
                out[c] = 0.f;
            }
        }
        return any;
    }

    // 4x4 cubic filters: interpolate each row along x, then along y
    float row[4][NC];
    for (int j = 0; j < 4; ++j) {
        float Ip[NC], Ic[NC], In[NC], Ia[NC];
        int yy = cy - 1 + j;
        any |= fetch<NC>(im, cx - 1, yy, sp.blackOutside, Ip);
        any |= fetch<NC>(im, cx,     yy, sp.blackOutside, Ic);
        any |= fetch<NC>(im, cx + 1, yy, sp.blackOutside, In);
        any |= fetch<NC>(im, cx + 2, yy, sp.blackOutside, Ia);
        for (int c = 0; c < NC; ++c) {
            row[j][c] = interp1(sp.filter, Ip[c], Ic[c], In[c], Ia[c], dx, sp.clamp);
        }
    }
    for (int c = 0; c < NC; ++c) {
        out[c] = interp1(sp.filter, row[0][c], row[1][c], row[2][c], row[3][c], dy, sp.clamp);
    }
    if (!any) {
        for (int c = 0; c < NC; ++c) {
            out[c] = 0.f;
        }
    }
    return any;
}

CPG_HD bool outsideBounds(const Pyramid& pyr, float x, float y)
{
    const FImage& im = pyr.level[0];
    float lx = x - pyr.x1, ly = y - pyr.y1;
    return (lx < 0.f) || (ly < 0.f) || (lx >= (float)im.width) || (ly >= (float)im.height);
}

// Box filter (as in Natron): integrate the source over the bounding box of the
// back-transformed pixel, outside is black or the closest pixel, normalized by the area.
template <int NC>
CPG_HD void boxFilter(const Pyramid& pyr, float fx, float fy,
                      float Jxx, float Jxy, float Jyx, float Jyy,
                      const SampleParams& sp, float* out)
{
    const FImage& im = pyr.level[0];
    const float hx = 0.5f * (fabsf(Jxx) + fabsf(Jxy));
    const float hy = 0.5f * (fabsf(Jyx) + fabsf(Jyy));
    if ( (hx <= 0.f) || (hy <= 0.f) ) {
        interpolate<NC>(pyr, fx, fy, sp, out); // empty pixel
        return;
    }
    const float x1 = fx - hx - pyr.x1, x2 = fx + hx - pyr.x1;
    const float y1 = fy - hy - pyr.y1, y2 = fy + hy - pyr.y1;
    const int i1 = (int)floorf(x1), i2 = (int)ceilf(x2) - 1;
    const int j1 = (int)floorf(y1), j2 = (int)ceilf(y2) - 1;
    // limit the cost: step over texels if the area is huge
    const int stepx = 1 + (i2 - i1) / CPG_BOX_MAX_TEXELS;
    const int stepy = 1 + (j2 - j1) / CPG_BOX_MAX_TEXELS;
    float acc[NC], t[NC];
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.f;
    }
    for (int j = j1; j <= j2; j += stepy) {
        const float wy = minf(y2, (float)(j + stepy)) - maxf(y1, (float)j);
        if (wy <= 0.f) {
            continue;
        }
        for (int i = i1; i <= i2; i += stepx) {
            const float wx = minf(x2, (float)(i + stepx)) - maxf(x1, (float)i);
            if (wx <= 0.f) {
                continue;
            }
            fetch<NC>(im, i, j, sp.blackOutside, t);
            for (int c = 0; c < NC; ++c) {
                acc[c] += wx * wy * t[c];
            }
        }
    }
    const float inv = 1.f / ( (x2 - x1) * (y2 - y1) );
    for (int c = 0; c < NC; ++c) {
        out[c] = acc[c] * inv;
    }
}

////////////////////////////////////////////////////////////////////////////////
// EWA (elliptical weighted average) filters over a mip pyramid.
//
// The Jacobian J of the inverse transform maps a destination pixel to an ellipse
// in the source. Its singular values (s1 >= s2) and directions give the ellipse
// axes. Each axis is widened to at least one source pixel (reconstruction), the
// mip level is chosen so that the minor axis stays small, and the major axis is
// capped (anisotropy limit) to keep the cost bounded.

CPG_HD float sincf_(float x)
{
    if (fabsf(x) < 1e-4f) {
        return 1.f;
    }
    const float px = 3.14159265358979f * x;
    return sinf(px) / px;
}

CPG_HD float lanczos3(float x)
{
    x = fabsf(x);
    return x < 3.f ? sincf_(x) * sincf_(x / 3.f) : 0.f;
}

template <int NC>
CPG_HD void ewaFilter(const Pyramid& pyr, float fx, float fy,
                      float Jxx, float Jxy, float Jyx, float Jyy,
                      const SampleParams& sp, float* out)
{
    // J J^T = [[a b][b c]]
    float a = Jxx * Jxx + Jxy * Jxy;
    float b = Jxx * Jyx + Jxy * Jyy;
    float c = Jyx * Jyx + Jyy * Jyy;
    float tr2 = 0.5f * (a + c);
    float disc = sqrtf( maxf(0.f, 0.25f * (a - c) * (a - c) + b * b) );
    float l1 = tr2 + disc;
    float l2 = maxf(0.f, tr2 - disc);
    float s1 = sqrtf(l1);
    float s2 = sqrtf(l2);
    // major axis direction
    float e1x, e1y;
    if (fabsf(b) > 1e-12f) {
        e1x = b;
        e1y = l1 - a;
        float n = sqrtf(e1x * e1x + e1y * e1y);
        e1x /= n; e1y /= n;
    } else if (a >= c) {
        e1x = 1.f; e1y = 0.f;
    } else {
        e1x = 0.f; e1y = 1.f;
    }
    float e2x = -e1y, e2y = e1x;
    // right singular vectors (destination axes): v_i = J^T e_i / s_i
    float v1x = 1.f, v1y = 0.f;
    if (s1 > 1e-8f) {
        v1x = (Jxx * e1x + Jyx * e1y) / s1;
        v1y = (Jxy * e1x + Jyy * e1y) / s1;
    }
    float v2x = -v1y, v2y = v1x;
    if (s2 > 1e-6f * s1) {
        v2x = (Jxx * e2x + Jyx * e2y) / s2;
        v2y = (Jxy * e2x + Jyy * e2y) / s2;
    }

    const bool gauss = (sp.filter == eEWAGaussian);
    // kernel scale along each axis, in level-0 texels
    float k1 = maxf(s1, 1.f);
    float k2 = maxf(s2, 1.f);
    // support radius in normalized kernel units
    const float sigma = 0.5f;                       // Gaussian sigma, in units of k
    const float R = gauss ? 3.f * sigma : 3.f;      // 3 sigma, or 3 lobes for Lanczos

    // choose the mip level: minor axis below ~2 texels, major axis below the cap
    int L = 0;
    {
        float minorR = R * k2, majorR = R * k1;
        while ( (L + 1 < pyr.nLevels) &&
                ( (minorR > 2.f * R) || (majorR > CPG_EWA_MAX_AXIS * R) ) ) {
            ++L;
            minorR *= 0.5f;
            majorR *= 0.5f;
        }
    }
    const float scale = 1.f / (float)(1 << L);
    const FImage& im = pyr.level[L];
    float lx = (fx - pyr.x1) * scale;
    float ly = (fy - pyr.y1) * scale;
    // singular values of the "clamped" Jacobian J' = U diag(q) V^T at this level:
    // at least one texel (reconstruction), at most the anisotropy cap.
    float q1 = clampf(s1 * scale, 1.f, CPG_EWA_MAX_AXIS);
    float q2 = clampf(s2 * scale, 1.f, q1);

    // The kernel is evaluated in the coordinates (u,v) = J'^-1 d, i.e. along the destination
    // pixel axes. J' is a continuous function of J, so the kernel never flips orientation, and
    // for a pure magnification (q1 = q2 = 1) it is the rotation of the destination grid.
    // Bounding box of |u|,|v| <= R: d = J' (u,v), J' = U diag(q) V^T
    const float B00 = q1 * e1x * v1x + q2 * e2x * v2x;
    const float B01 = q1 * e1x * v1y + q2 * e2x * v2y;
    const float B10 = q1 * e1y * v1x + q2 * e2y * v2x;
    const float B11 = q1 * e1y * v1y + q2 * e2y * v2y;
    float hx = R * (fabsf(B00) + fabsf(B01));
    float hy = R * (fabsf(B10) + fabsf(B11));
    int i1 = (int)floorf(lx - hx - 0.5f), i2 = (int)ceilf(lx + hx - 0.5f);
    int j1 = (int)floorf(ly - hy - 0.5f), j2 = (int)ceilf(ly + hy - 0.5f);
    const float iq1 = 1.f / q1, iq2 = 1.f / q2;
    const float inv2s2 = 1.f / (2.f * sigma * sigma);
    // A = J'^-1 = V diag(1/q) U^T
    const float A00 = v1x * e1x * iq1 + v2x * e2x * iq2;
    const float A01 = v1x * e1y * iq1 + v2x * e2y * iq2;
    const float A10 = v1y * e1x * iq1 + v2y * e2x * iq2;
    const float A11 = v1y * e1y * iq1 + v2y * e2y * iq2;

    float acc[NC], vmin[NC], vmax[NC], t[NC];
    for (int ch = 0; ch < NC; ++ch) {
        acc[ch] = 0.f;
        vmin[ch] = 3.4e38f;
        vmax[ch] = -3.4e38f;
    }
    float wsum = 0.f;
    for (int j = j1; j <= j2; ++j) {
        float dy = (j + 0.5f) - ly;
        for (int i = i1; i <= i2; ++i) {
            float dx = (i + 0.5f) - lx;
            float u = A00 * dx + A01 * dy;
            float v = A10 * dx + A11 * dy;
            float w;
            if (gauss) {
                float r2 = u * u + v * v;
                if (r2 > R * R) {
                    continue;
                }
                w = expf(-r2 * inv2s2);
            } else {
                if ( (fabsf(u) >= R) || (fabsf(v) >= R) ) {
                    continue;
                }
                w = lanczos3(u) * lanczos3(v);
            }
            if (w == 0.f) {
                continue;
            }
            fetch<NC>(im, i, j, sp.blackOutside, t);
            for (int ch = 0; ch < NC; ++ch) {
                acc[ch] += w * t[ch];
                if (w > 0.f) {
                    vmin[ch] = minf(vmin[ch], t[ch]);
                    vmax[ch] = maxf(vmax[ch], t[ch]);
                }
            }
            wsum += w;
        }
    }
    for (int ch = 0; ch < NC; ++ch) {
        float v = (wsum != 0.f) ? acc[ch] / wsum : 0.f;
        if (sp.clamp && !gauss && (vmax[ch] >= vmin[ch])) {
            v = clampf(v, vmin[ch], vmax[ch]); // anti-ringing
        }
        out[ch] = v;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sample the source for destination pixel center (px,py) through inverse transform H.

template <int NC>
CPG_HD void sampleTransformed(const Pyramid& pyr, const Mat3& Hm, float px, float py,
                              const SampleParams& sp, float* out)
{
    const float* H = Hm.m;
    float X = H[0] * px + H[1] * py + H[2];
    float Y = H[3] * px + H[4] * py + H[5];
    float Z = H[6] * px + H[7] * py + H[8];

    if ( (Z <= 0.f) || (pyr.level[0].width <= 0) ) {
        // the back-transformed point is at infinity (==0) or behind the camera (<0)
        for (int c = 0; c < NC; ++c) {
            out[c] = 0.f;
        }
        return;
    }
    const float iz = 1.f / Z;
    const float fx = X * iz;
    const float fy = Y * iz;

    if (sp.filter == eImpulse) {
        interpolate<NC>(pyr, fx, fy, sp, out);
        return;
    }

    // Jacobian of the projective mapping
    const float iz2 = iz * iz;
    float Jxx = (H[0] * Z - X * H[6]) * iz2;
    float Jxy = (H[1] * Z - X * H[7]) * iz2;
    float Jyx = (H[3] * Z - Y * H[6]) * iz2;
    float Jyy = (H[4] * Z - Y * H[7]) * iz2;

    if ( isEWA(sp.filter) ) {
        // quick reject: the whole footprint maps outside the source and outside is black
        if (sp.blackOutside) {
            const FImage& im = pyr.level[0];
            const float hx = 3.f * (0.5f * (fabsf(Jxx) + fabsf(Jxy)) + 2.f) + 3.f;
            const float hy = 3.f * (0.5f * (fabsf(Jyx) + fabsf(Jyy)) + 2.f) + 3.f;
            const float lx = fx - pyr.x1, ly = fy - pyr.y1;
            if ( (lx + hx < 0.f) || (ly + hy < 0.f) || (lx - hx > im.width) || (ly - hy > im.height) ) {
                for (int c = 0; c < NC; ++c) {
                    out[c] = 0.f;
                }
                return;
            }
        }
        ewaFilter<NC>(pyr, fx, fy, Jxx, Jxy, Jyx, Jyy, sp, out);
        return;
    }

    // Classic filters: same algorithm as Natron (Transform3x3Processor + ofxsFilterInterpolate2DSuper)
    {
        // no supersampling outside of the source
        const FImage& im = pyr.level[0];
        const float lx = fx - pyr.x1, ly = fy - pyr.y1;
        bool xinside = (0.f <= lx + 0.5f) && (lx - 0.5f < (float)im.width);
        bool yinside = (0.f <= ly + 0.5f) && (ly - 0.5f < (float)im.height);
        if ( sp.blackOutside && !(xinside && yinside) ) {
            xinside = yinside = false;
        }
        if (!xinside) {
            Jxx = Jxy = 0.f;
        }
        if (!yinside) {
            Jyx = Jyy = 0.f;
        }
    }
    if ( (Jxx == 0.f) && (Jxy == 0.f) && (Jyx == 0.f) && (Jyy == 0.f) ) {
        interpolate<NC>(pyr, fx, fy, sp, out);
        return;
    }
    if (sp.filter == eBox) {
        boxFilter<NC>(pyr, fx, fy, Jxx, Jxy, Jyx, Jyy, sp, out);
        return;
    }

    // center value, with the requested filter
    bool inside = interpolate<NC>(pyr, fx, fy, sp, out);
    if (!inside) {
        // the center is outside and black: stop if the whole pixel is outside
        if ( outsideBounds(pyr, fx - Jxx * 0.5f - Jxy * 0.5f, fy - Jyx * 0.5f - Jyy * 0.5f) &&
             outsideBounds(pyr, fx + Jxx * 0.5f - Jxy * 0.5f, fy + Jyx * 0.5f - Jyy * 0.5f) &&
             outsideBounds(pyr, fx - Jxx * 0.5f + Jxy * 0.5f, fy - Jyx * 0.5f + Jyy * 0.5f) &&
             outsideBounds(pyr, fx + Jxx * 0.5f + Jxy * 0.5f, fy + Jyx * 0.5f + Jyy * 0.5f) ) {
            return;
        }
    }
    const float dx = Jxx * Jxx + Jyx * Jyx; // squared length of the back-projected pixel along x
    const float dy = Jxy * Jxy + Jyy * Jyy;
    if ( (dx <= 1.f) && (dy <= 1.f) ) {
        return; // no minification
    }
    // supersampling on a 3^n x 3^n grid (n <= 4), the other samples are bilinear
    const float log9 = 2.f * 1.0986122886681098f; // 2 log(3)
    const float sx = (dx <= 1.f) ? 0.f : minf(logf(dx) / log9, 4.f);
    const float sy = (dy <= 1.f) ? 0.f : minf(logf(dy) / log9, 4.f);
    const int isx = (int)ceilf(sx - 0.5f);
    const int isy = (int)ceilf(sy - 0.5f);
    int nx = 1, ny = 1;
    for (int k = 0; k < isx; ++k) {
        nx *= 3;
    }
    for (int k = 0; k < isy; ++k) {
        ny *= 3;
    }
    SampleParams bil = sp;
    bil.filter = eBilinear;
    bil.clamp = false;
    float acc[NC], t[NC];
    for (int c = 0; c < NC; ++c) {
        acc[c] = out[c];
    }
    const float inx = 1.f / nx, iny = 1.f / ny;
    for (int y = -ny / 2; y <= ny / 2; ++y) {
        for (int x = -nx / 2; x <= nx / 2; ++x) {
            if ( (x == 0) && (y == 0) ) {
                continue; // center already accumulated
            }
            const float sfx = fx + Jxx * x * inx + Jxy * y * iny;
            const float sfy = fy + Jyx * x * inx + Jyy * y * iny;
            interpolate<NC>(pyr, sfx, sfy, bil, t);
            for (int c = 0; c < NC; ++c) {
                acc[c] += t[c];
            }
        }
    }
    const float inv = 1.f / (nx * ny);
    for (int c = 0; c < NC; ++c) {
        out[c] = acc[c] * inv;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Motion blur: same adaptive Monte-Carlo integration as Natron's Transform3x3
// (at least 13 stratified samples, then van der Corput samples until the
// variance of the mean is small enough or the maximum count is reached).

CPG_HD unsigned int hashu(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

CPG_HD float vanDerCorput2(unsigned int seed)
{
    // bit reversal of a 32-bit integer gives the base-2 radical inverse
#if defined(__CUDA_ARCH__)
    return (float)__brev(seed) * 2.3283064365386963e-10f;
#else
    seed = (seed << 16) | (seed >> 16);
    seed = ( (seed & 0x00ff00ffu) << 8 ) | ( (seed & 0xff00ff00u) >> 8 );
    seed = ( (seed & 0x0f0f0f0fu) << 4 ) | ( (seed & 0xf0f0f0f0u) >> 4 );
    seed = ( (seed & 0x33333333u) << 2 ) | ( (seed & 0xccccccccu) >> 2 );
    seed = ( (seed & 0x55555555u) << 1 ) | ( (seed & 0xaaaaaaaau) >> 1 );
    return (float)seed * 2.3283064365386963e-10f;
#endif
}

struct BlurParams
{
    const Mat3* H;       // inverse transforms (pixel coordinates)
    const float* alpha;  // per-transform weight, or NULL
    int count;           // number of transforms
    float motionblur;    // quality, 0 = disabled
    float maxValue;      // 1, 255 or 65535
};

template <int NC>
CPG_HD void renderPixel(const Pyramid& pyr, int x, int y,
                        const SampleParams& sp, const BlurParams& bp, float* out)
{
    const float px = x + 0.5f, py = y + 0.5f;
    if ( (bp.motionblur <= 0.f) || (bp.count <= 1) ) {
        sampleTransformed<NC>(pyr, bp.H[0], px, py, sp, out);
        return;
    }
    const float maxErr = bp.motionblur * bp.maxValue / 1000.f;
    const float maxErr2 = maxErr * maxErr;
    const int maxIt = (int)(bp.motionblur * 40);
    const int minsamples = maxIt / 3 > 13 ? maxIt / 3 : 13;
    float accPix[NC], accPix2[NC], mean[NC], t[NC];
    for (int c = 0; c < NC; ++c) {
        accPix[c] = accPix2[c] = mean[c] = 0.f;
    }
    float acc = 0.f;
    unsigned int seed = hashu( hashu( (unsigned int)x + (unsigned int)(0x10000 * bp.motionblur) ) + (unsigned int)y );
    int sample = 0;
    int maxsamples = minsamples;
    while (sample < maxsamples) {
        for (; sample < maxsamples; ++sample, ++seed) {
            int ti;
            if (sample < minsamples) {
                ti = (int)( (sample + vanDerCorput2(seed)) * bp.count / (float)minsamples );
            } else {
                ti = (int)(vanDerCorput2(seed) * bp.count);
            }
            ti = clampi(ti, 0, bp.count - 1);
            sampleTransformed<NC>(pyr, bp.H[ti], px, py, sp, t);
            const float w = bp.alpha ? bp.alpha[ti] : 1.f;
            acc += w;
            for (int c = 0; c < NC; ++c) {
                accPix[c] += t[c] * w;
                accPix2[c] += t[c] * t[c] * w;
            }
        }
        if (acc > 0.f) {
            for (int c = 0; c < NC; ++c) {
                mean[c] = accPix[c] / acc;
                if (sample > 1) {
                    float var = bp.alpha ? (accPix2[c] / acc - mean[c] * mean[c])
                                         : (accPix2[c] - mean[c] * mean[c] * sample) / (sample - 1);
                    if (maxsamples < maxIt) {
                        int want = (int)(var / maxErr2);
                        want = want < maxIt ? want : maxIt;
                        maxsamples = want > maxsamples ? want : maxsamples;
                    }
                }
            }
        }
    }
    for (int c = 0; c < NC; ++c) {
        out[c] = mean[c];
    }
}

// Mask / mix with the untransformed source (same as ofxsMaskMixPix). srcPix may be NULL.
template <int NC>
CPG_HD void maskMix(const float* tmpPix, const float* srcPix, float maskAlpha, float* out)
{
    for (int c = 0; c < NC; ++c) {
        float s = srcPix ? srcPix[c] : 0.f;
        out[c] = (maskAlpha == 1.f) ? tmpPix[c] : (tmpPix[c] * maskAlpha + (1.f - maskAlpha) * s);
    }
}

// Downsample one pyramid level by 2 (box filter, edge texels clamped).
template <int NC>
CPG_HD void downsamplePixel(const FImage& src, float* dst, long long dstStride, int i, int j)
{
    float acc[NC];
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.f;
    }
    for (int dj = 0; dj < 2; ++dj) {
        int sj = clampi(2 * j + dj, 0, src.height - 1);
        for (int di = 0; di < 2; ++di) {
            int si = clampi(2 * i + di, 0, src.width - 1);
            const float* p = src.data + (long long)sj * src.stride + (long long)si * NC;
            for (int c = 0; c < NC; ++c) {
                acc[c] += p[c];
            }
        }
    }
    float* d = dst + (long long)j * dstStride + (long long)i * NC;
    for (int c = 0; c < NC; ++c) {
        d[c] = 0.25f * acc[c];
    }
}
} // namespace cpg

#endif // CORNERPINEX_OFX_SAMPLER_H
