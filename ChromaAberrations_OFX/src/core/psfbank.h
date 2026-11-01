// ChromaAberrations_OFX - bank of PSFs over a field grid and a set of wavelengths
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <functional>
#include <vector>

#include "psf.h"
#include "spectrum.h"

namespace ca {

struct ImageGeometry
{
    int width = 2048;              // image size in pixels
    int height = 1152;
    double filmbackWidthMm = 24.89; // sensor width covered by the image (Super 35 by default)
    double pixelAspect = 1.;       // > 1 for anamorphic plates (pixels wider than tall): the
                                   // filmback is the squeezed width on the sensor, the lens is
                                   // modelled in the desqueezed space
    double centerX = 0.5, centerY = 0.5; // optical centre, as a fraction of the image size

    double pixelPitchUm() const { return filmbackWidthMm * 1000. / width; }
};

struct PsfBank
{
    int gridX = 0, gridY = 0;
    std::vector<WavelengthSample> wavelengths;
    // psf[(gy * gridX + gx) * nLambda + l]
    std::vector<PsfResult> psf;
    std::vector<double> fieldX, fieldY;    // normalised field point of each grid cell
    std::vector<double> pixelX, pixelY;    // image position (pixels) of each grid cell

    const PsfResult& at(int gx, int gy, int l) const { return psf[( (size_t)gy * gridX + gx ) * wavelengths.size() + l]; }
};

// Field point (normalised, |H| = 1 at the farthest image corner) of an image position.
void pixelToField(const ImageGeometry& g, double px, double py, double& hx, double& hy);

// Computes every PSF of the grid on all CPU cores. 'progress' (optional) is called with the
// number of PSFs done and the total.
// Parametric lens (steps 1-3)
PsfBank computePsfBank(const WavefrontParams& w, const ApertureParams& a, const ImageGeometry& g,
                       int kernelSize, int gridX, int gridY, int nWavelengths, const PsfOptions& opt,
                       std::function<void(int, int)> progress = std::function<void(int, int)>());

// Any lens model, already prepared for these wavelengths (see lensmodel.h)
class LensModel;

// PSF of a model at image position (px, py) (pixels) and render wavelength l (nm), on plate pixels
PsfResult computePsfAt(const LensModel& model, const ImageGeometry& g, int kernelSize, double px, double py,
                       int l, double nm, const PsfOptions& opt);
// Radius (plate pixels) that holds the PSF there: geometric spot + diffraction margin
double psfExtentPx(const LensModel& model, const ImageGeometry& g, double px, double py, int l, double nm);

PsfBank computePsfBank(const LensModel& model, const ImageGeometry& g, int kernelSize, int gridX, int gridY,
                       const std::vector<WavelengthSample>& wavelengths, const PsfOptions& opt,
                       std::function<void(int, int)> progress = std::function<void(int, int)>());

} // namespace ca
