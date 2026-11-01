// ChromaAberrations_OFX - RGB <-> spectrum conversion for the renderer
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// The renderer works wavelength by wavelength. Each RGB pixel is turned into a spectrum
// sampled at the render wavelengths, and the light of each wavelength is turned back into
// RGB with the CIE 1931 colour matching functions expressed in the working space, or with the
// sensitivities of a camera carried into the working space (camera.h).
//
// Each colour gets the smoothest spectrum (least second differences, like natural spectra)
// that has exactly its RGB values and no negative light where that is possible:
//   - RGB -> spectrum -> RGB is the identity, exactly, for any value (HDR, negative);
//   - greys become flat spectra, so a white highlight disperses into a real rainbow;
//   - a saturated light (a red neon in ACEScg, say) gets a narrow, positive spectrum instead
//     of negative lobes that would show as dark fringes once the lens separates wavelengths.
// The spectrum is proportional to the intensity (scaling a colour scales its spectrum), so it
// is stored for colours of unit sum r + g + b on a chromaticity grid (lookup table) and
// interpolated bilinearly. Interpolation keeps the round trip exact: the RGB of the nodes is an
// affine function of the grid coordinates, which bilinear interpolation reproduces exactly.
// Colours outside the table (or of zero or negative sum, i.e. noise in the blacks) use the
// linear basis: spectrum = sum_d B_d rgb_d, also exact.

#pragma once

#include <string>
#include <vector>

#include "camera.h"
#include "spectrum.h"

namespace ca {

enum ColorSpace
{
    eColorSpaceACEScg = 0,   // ACES AP1 primaries, linear (default)
    eColorSpaceRec709,       // linear sRGB / Rec.709 primaries
    eColorSpaceRec2020,      // linear Rec.2020 primaries
};

const char* colorSpaceName(ColorSpace cs);
bool colorSpaceFromName(const std::string& name, ColorSpace& cs);

// 3 x 3 matrix (row-major) converting linear RGB of the space to CIE XYZ
void rgbToXyzMatrix(ColorSpace cs, double m[9]);

struct SpectralBasis
{
    ColorSpace space = eColorSpaceACEScg;
    std::vector<WavelengthSample> wavelengths;
    std::vector<double> basis[3];     // B_d(i): spectrum of a unit R, G or B value
    std::vector<double> response[3];  // w_c(i): contribution of wavelength i to output channel c

    // chromaticity table: node (i, j) is the colour r = lutLo + i * lutStep, g = lutLo + j * lutStep,
    // b = 1 - r - g; lut[(l * lutN + j) * lutN + i] is its spectrum at wavelength l
    int lutN = 0;
    double lutLo = 0., lutStep = 1.;
    std::vector<float> lut;

    // camera (empty: CIE 1931 observer): matrix from the camera's RGB (each channel normalised
    // to 1 for a flat spectrum) to the working space, and its fit error on the training colours
    // (RMS, relative to white)
    std::string camera;
    double cameraMatrix[9] = { 1., 0., 0., 0., 1., 0., 0., 0., 1. };
    double cameraFitError = 0.;

    int count() const { return (int)wavelengths.size(); }
};

SpectralBasis buildSpectralBasis(int nWavelengths, ColorSpace cs, double loNm = 400., double hiNm = 700.,
                                 const CameraSensitivity* camera = 0);

// Matrix from camera RGB to the working space (see camera.h), fitted between loNm and hiNm.
// Returns the RMS error on the training colours, relative to white.
double fitCameraMatrix(const CameraSensitivity& cam, ColorSpace cs, double loNm, double hiNm, double m[9]);

// Spectrum of one colour at wavelength l (as used by the renderers)
double spectrumOf(const SpectralBasis& sb, int l, double r, double g, double b);

} // namespace ca
