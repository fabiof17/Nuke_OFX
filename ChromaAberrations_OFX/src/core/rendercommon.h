// ChromaAberrations_OFX - per-pixel math shared by the CPU and CUDA renderers
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Both renderers call these exact functions (compiled for the host and the device), so the
// GPU result can only differ from the CPU reference by float rounding and FFT precision.
//
// Coordinates: images are stored bottom-up (OFX convention), pixel (i, j) has its centre at
// (i + 0.5, j + 0.5). The renderer works on a padded domain of (W + 2R) x (H + 2R) pixels:
// padded pixel (i, j) is image pixel (i - R, j - R).

#pragma once

#ifdef __CUDACC__
#define CA_HD __host__ __device__ __forceinline__
#else
#define CA_HD inline
#endif

namespace ca {

struct PixelMath
{
    // image
    int width, height;
    int R;                     // padding (kernel radius)
    // maps
    int mapW, mapH;
    float mapX0, mapY0, mapInvStep;
    // PSF grid
    int gridX, gridY;
    // spectral table (see spectralbasis.h)
    int lutN;
    float lutLo, lutInvStep;
    float gridSx, gridSy;      // (gridX - 1) / width, (gridY - 1) / height
};

CA_HD float caClamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// bilinear lookup in a map of nodes (mapW x mapH), clamped at the borders
CA_HD float mapLookup(const float* m, const PixelMath& pm, float x, float y)
{
    float u = caClamp( (x - pm.mapX0) * pm.mapInvStep, 0.f, (float)(pm.mapW - 1) );
    float v = caClamp( (y - pm.mapY0) * pm.mapInvStep, 0.f, (float)(pm.mapH - 1) );
    int i = (int)u, j = (int)v;
    if (i > pm.mapW - 2) i = pm.mapW - 2 < 0 ? 0 : pm.mapW - 2;
    if (j > pm.mapH - 2) j = pm.mapH - 2 < 0 ? 0 : pm.mapH - 2;
    const float fu = u - i, fv = v - j;
    const int i1 = i + 1 < pm.mapW ? i + 1 : i, j1 = j + 1 < pm.mapH ? j + 1 : j;
    const float a = m[j * pm.mapW + i], b = m[j * pm.mapW + i1];
    const float c = m[j1 * pm.mapW + i], d = m[j1 * pm.mapW + i1];
    return (1.f - fv) * ( (1.f - fu) * a + fu * b ) + fv * ( (1.f - fu) * c + fu * d );
}

// bilinear sample of an image plane at a continuous position, edge pixels extended
CA_HD float sampleImage(const float* img, int w, int h, float x, float y)
{
    float u = caClamp(x - 0.5f, 0.f, (float)(w - 1));
    float v = caClamp(y - 0.5f, 0.f, (float)(h - 1));
    int i = (int)u, j = (int)v;
    if (i > w - 2) i = w - 2 < 0 ? 0 : w - 2;
    if (j > h - 2) j = h - 2 < 0 ? 0 : h - 2;
    const float fu = u - i, fv = v - j;
    const int i1 = i + 1 < w ? i + 1 : i, j1 = j + 1 < h ? j + 1 : j;
    const float a = img[(size_t)j * w + i], b = img[(size_t)j * w + i1];
    const float c = img[(size_t)j1 * w + i], d = img[(size_t)j1 * w + i1];
    return (1.f - fv) * ( (1.f - fu) * a + fu * b ) + fv * ( (1.f - fu) * c + fu * d );
}

// Spectrum of a colour at one wavelength: chromaticity table 'lut' (the plane of that
// wavelength), or the linear basis (br, bg, bb) outside the table. Same as spectrumOf().
CA_HD float spectralValue(const PixelMath& pm, const float* lut, float br, float bg, float bb, float r, float g, float b)
{
    const float sum = r + g + b;
    const float mag = (r < 0.f ? -r : r) + (g < 0.f ? -g : g) + (b < 0.f ? -b : b);
    if ( lut && (pm.lutN > 1) && (sum > 0.f) && (sum > 1e-6f * mag) ) {
        const float inv = 1.f / sum;
        const float u = (r * inv - pm.lutLo) * pm.lutInvStep, v = (g * inv - pm.lutLo) * pm.lutInvStep;
        const float last = (float)(pm.lutN - 1);
        if ( (u >= 0.f) && (v >= 0.f) && (u <= last) && (v <= last) ) {
            int i = (int)u, j = (int)v;
            if (i > pm.lutN - 2) i = pm.lutN - 2;
            if (j > pm.lutN - 2) j = pm.lutN - 2;
            const float fu = u - i, fv = v - j;
            const float a = lut[j * pm.lutN + i], c1 = lut[j * pm.lutN + i + 1];
            const float c2 = lut[(j + 1) * pm.lutN + i], c3 = lut[(j + 1) * pm.lutN + i + 1];
            return sum * ( (1.f - fv) * ( (1.f - fu) * a + fu * c1 ) + fv * ( (1.f - fu) * c2 + fu * c3 ) );
        }
    }
    return br * r + bg * g + bb * b;
}

// Light of one wavelength arriving at padded pixel (i, j): the image point whose chief ray
// lands there (inverse displacement map of the plan), converted to the wavelength's spectral
// value and attenuated by the relative illumination.
CA_HD float sourceValue(const PixelMath& pm, int i, int j,
                        const float* r, const float* g, const float* b,
                        const float* dispX, const float* dispY, const float* illum,
                        const float* lut, float br, float bg, float bb)
{
    const float x = i - pm.R + 0.5f, y = j - pm.R + 0.5f;
    float xs = x, ys = y;
    if (dispX) {
        xs = x - mapLookup(dispX, pm, x, y);
        ys = y - mapLookup(dispY, pm, x, y);
    }
    const float s = spectralValue(pm, lut, br, bg, bb,
                                  sampleImage(r, pm.width, pm.height, xs, ys),
                                  sampleImage(g, pm.width, pm.height, xs, ys),
                                  sampleImage(b, pm.width, pm.height, xs, ys));
    return illum ? s * mapLookup(illum, pm, x, y) : s;
}

// Bilinear weight of grid point (gx, gy) at padded pixel (i, j). The weights of all grid
// points sum to 1 everywhere (beyond the outer grid points the border ones take it all).
CA_HD float nodeWeight(const PixelMath& pm, int gx, int gy, int i, int j)
{
    float wx = 1.f, wy = 1.f;
    if (pm.gridX > 1) {
        const float u = caClamp( (i - pm.R + 0.5f) * pm.gridSx, 0.f, (float)(pm.gridX - 1) );
        wx = 1.f - (u > gx ? u - gx : gx - u);
        wx = wx > 0.f ? wx : 0.f;
    }
    if (pm.gridY > 1) {
        const float v = caClamp( (j - pm.R + 0.5f) * pm.gridSy, 0.f, (float)(pm.gridY - 1) );
        wy = 1.f - (v > gy ? v - gy : gy - v);
        wy = wy > 0.f ? wy : 0.f;
    }
    return wx * wy;
}

} // namespace ca
