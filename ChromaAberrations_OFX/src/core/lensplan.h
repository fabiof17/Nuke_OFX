// ChromaAberrations_OFX - everything the renderer needs from the lens, computed once per setting
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// The lens plan depends only on the optical settings and the image size, never on the image
// content: it is computed once and reused for every frame (and uploaded once to the GPU).
//
//  - spectral basis: RGB -> spectrum -> RGB for the render wavelengths;
//  - per wavelength, the PSF bank: a grid of PSFs centred on the chief ray, normalised to 1.
//    The light landing at x is spread by the PSF interpolated bilinearly between the 4 grid
//    PSFs around x. Since the bilinear weights w_g(x) of the grid points g sum to 1, this
//    blur that varies over the image is a sum of ordinary convolutions:
//        out = sum_g (w_g s) * PSF_g           (* = convolution, s = light of that wavelength)
//    and w_g is zero beyond the neighbouring grid points, so each term is a convolution of a
//    small tile (2 x 2 grid cells + the PSF margin) done by FFT and added back in place.
//    The result is exact (no approximation of the PSFs, never negative light);
//  - per wavelength, the chief ray displacement map (distortion + lateral colour), inverted:
//    for each landing position, where the light comes from;
//  - the relative illumination map (natural cos^4 falloff and mechanical vignetting).

#pragma once

#include <memory>

#include <functional>
#include <string>
#include <vector>

#include "psfbank.h"
#include "spectralbasis.h"

namespace ca {

struct RenderSettings
{
    int wavelengths = 8;          // render wavelengths between 400 and 700 nm
    int gridX = 7, gridY = 5;     // PSF grid over the image
    int kernelSize = 65;          // PSF size in pixels (odd); with autoKernel, the largest allowed
    bool autoKernel = false;      // size the PSFs from the lens (smaller and faster when it is sharp)
    bool autoGrid = false;        // refine the grid until the interpolation error is below gridTolerance
    double gridTolerance = 0.02;  // L1 error of the PSF interpolated in the middle of a grid cell
    int gridMax = 25;             // largest automatic grid (per axis)
    ColorSpace colorSpace = eColorSpaceACEScg;
    std::shared_ptr<const CameraSensitivity> camera;   // null: CIE 1931 observer

    bool blur = true;             // aberrations and diffraction (PSF)
    bool distortion = true;       // chief ray displacement: distortion and lateral colour
    bool vignetting = true;       // relative illumination

    double focalLengthMm = 35.;   // for the natural (cos^4) falloff
    double naturalVignetting = 1.;// 0 = none, 1 = physical cos^4 law
    int mapStep = 16;             // spacing of the displacement / illumination maps (pixels)

    PsfOptions psf;
};

struct LensPlan
{
    ImageGeometry geom;
    RenderSettings settings;
    SpectralBasis spectral;

    int S = 1, R = 0;             // kernel size and radius (pixels)
    int gridX = 1, gridY = 1;

    // per wavelength: the grid PSFs, normalised to 1, S x S floats each, rows bottom-up,
    // [(gy * gridX + gx) * S * S + ...]
    std::vector<std::vector<float> > bankKernels;
    double minEnergyInKernel = 1.; // smallest share of a PSF's light inside the kernel (before normalisation)

    // displacement and illumination maps: node (i, j) sits at pixel position
    // (mapX0 + i * mapStep, mapY0 + j * mapStep) (pixel centres are at k + 0.5)
    int mapW = 0, mapH = 0;
    double mapX0 = 0., mapY0 = 0., mapStep = 16.;
    // per wavelength, pixels: inverse displacement u(x) = x - x_s, where x_s is the image point
    // whose chief ray lands at x (distortion and lateral colour)
    std::vector<std::vector<float> > dispX, dispY;
    std::vector<float> illum;
    double maxDisplacementPx = 0.;

    double gridError = -1.;       // estimated interpolation error (automatic grid), -1: not measured
    double buildSeconds = 0.;     // total, and its parts:
    double prepareSeconds = 0.;   //   lens model (ray tracing of a real lens)
    double psfSeconds = 0.;       //   PSFs (and the automatic grid / size)
    double mapsSeconds = 0.;      //   distortion and illumination maps

    int wavelengthCount() const { return spectral.count(); }
    int nodeCount() const { return gridX * gridY; }
    // image position (pixels) of grid column gx / row gy: the grid reaches the image borders
    double nodeX(int gx) const { return gridX > 1 ? (double)geom.width * gx / (gridX - 1) : 0.5 * geom.width; }
    double nodeY(int gy) const { return gridY > 1 ? (double)geom.height * gy / (gridY - 1) : 0.5 * geom.height; }
    // continuous grid coordinates of an image position (pixels), clamped to the grid
    void gridCoords(double x, double y, double& u, double& v) const;
};

// Parametric lens
LensPlan buildLensPlan(const WavefrontParams& w, const ApertureParams& a, const ImageGeometry& g,
                       const RenderSettings& rs,
                       std::function<void(int, int)> progress = std::function<void(int, int)>());

// Any lens model (parametric or real, see lensmodel.h); false if the model cannot be prepared.
// focalLengthMm and naturalVignetting of the settings: cos^4 falloff of models that do not
// include it, or amount of the model's own relative illumination.
class LensModel;
bool buildLensPlan(LensModel& model, const ImageGeometry& g, const RenderSettings& rs, LensPlan& plan,
                   std::function<void(int, int)> progress = std::function<void(int, int)>(), std::string* error = 0);

} // namespace ca
