// ChromaAberrations_OFX - image renderer (CPU reference) and the exact renderer for the tests
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <string>
#include <vector>

#include "lensplan.h"
#include "rendercommon.h"

namespace ca {

// Planar linear RGB image, rows bottom-up (OFX convention)
struct ImageRGB
{
    int width = 0, height = 0;
    std::vector<float> c[3];

    void resize(int w, int h)
    {
        width = w;
        height = h;
        for (int k = 0; k < 3; ++k) {
            c[k].assign( (size_t)w * h, 0.f );
        }
    }
};

struct RenderStats
{
    int fftWidth = 0, fftHeight = 0;   // transform size of one tile
    int convolutions = 0;              // tile convolutions (grid points x wavelengths)
    double seconds = 0.;               // whole render
    double sourceSeconds = 0.;         // spectral conversion, distortion, illumination
    double fftSeconds = 0.;            // transforms and spectral products (GPU: all GPU work)
    double transferSeconds = 0.;       // GPU: uploads and downloads
    bool kernelCache = false;          // GPU: PSF spectra kept from a previous frame
    int batches = 1;                   // GPU: wavelength batches (more than 1 when the memory is short)
    std::string device;                // "CPU" or the GPU name
};

// Smallest size >= n whose prime factors are 2, 3, 5 and 7 (fast for pocketfft and cuFFT)
int goodFftSize(int n);

PixelMath makePixelMath(const LensPlan& plan);

// Tiles of the padded domain (W + 2R) x (H + 2R), one per grid point: the source pixels where
// the grid point's bilinear weight is not zero ([i0, i1] x [j0, j1], padded coordinates), and
// the tile (origin ox, oy) that also holds the PSF margin. All tiles share one FFT size.
struct TileLayout
{
    int dw = 0, dh = 0;            // padded domain
    int tw = 0, th = 0;            // FFT size of a tile
    std::vector<int> i0, i1, j0, j1, ox, oy;   // per grid point (gy * gridX + gx)
};
TileLayout makeTileLayout(const LensPlan& plan);

// The renderer: per grid point and wavelength, the weighted tile of light convolved by the
// grid PSF (FFT), accumulated over wavelengths into the three channels in the frequency
// domain, then added back into the image.
void renderCpu(const LensPlan& plan, const ImageRGB& in, ImageRGB& out, RenderStats* stats = 0);

// Reference: every source pixel is spread with the PSF interpolated bilinearly between the
// grid PSFs (direct, no FFT). Slow: for tests on small images.
void renderExactCpu(const LensPlan& plan, const ImageRGB& in, ImageRGB& out);

} // namespace ca
