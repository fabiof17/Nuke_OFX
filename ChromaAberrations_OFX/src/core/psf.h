// ChromaAberrations_OFX - point spread function engine
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <string>
#include <vector>

#include "aperture.h"
#include "wavefront.h"

namespace ca {

struct SensorParams
{
    double pixelPitchUm = 5.;   // size of one image pixel on the sensor (micrometres)
    int kernelSize = 65;        // PSF kernel size in pixels (odd)
};

enum PsfMethod
{
    eMethodAuto = 0,   // wave optics when the sampling allows it, else hybrid
    eMethodWave,       // always wave optics (FFT of the pupil function)
    eMethodGeometric   // always hybrid: ray spot convolved with the diffraction PSF
};

struct PsfOptions
{
    PsfMethod method = eMethodAuto;
    int maxFFT = 1024;          // largest FFT size for the wave method (beyond: hybrid, which agrees
                                // within a few % at that level of aberration and is much faster)
    int geoSamples = 640;       // pupil samples per axis for the ray spot
};

struct PsfResult
{
    int size = 0;               // kernel size (odd), centred on the chief ray
    std::vector<float> kernel;  // size * size, row-major, y up; sums to the energy inside the kernel
    double energyInKernel = 0.; // fraction of the light that falls inside the kernel
    double dispXPx = 0., dispYPx = 0.;   // chief ray displacement (distortion + lateral colour), pixels
    double centroidXPx = 0., centroidYPx = 0.; // centroid of the kernel relative to its centre, pixels
    double ee80Px = -1.;        // radius around the centre holding 80 % of the light (-1: not in the kernel)
    std::string method;         // "wave" or "hybrid"
    int pupilSamples = 0;       // K: pupil samples across the diameter (wave method)
    int fftSize = 0;            // M: FFT size (wave method)
};

// PSF of the lens at field point (hx, hy) (normalised, |H| = 1 at the image corner) and
// wavelength lambdaNm, integrated over the sensor pixels.
PsfResult computePsf(const WavefrontParams& w, const ApertureParams& a, const SensorParams& s,
                     double hx, double hy, double lambdaNm, const PsfOptions& opt = PsfOptions());

// Encircled energy radius (pixels) around the kernel centre for a given fraction of the
// total light (1 = everything, including what falls outside the kernel). -1 if not reached.
// Radius of the geometric spot (largest transverse aberration over the pupil), micrometres
double psfGeometricRadiusUm(const WavefrontParams& w, const ApertureParams& a, double hx, double hy, double lambdaNm);

double encircledEnergyRadius(const std::vector<float>& kernel, int size, double fraction);

} // namespace ca
