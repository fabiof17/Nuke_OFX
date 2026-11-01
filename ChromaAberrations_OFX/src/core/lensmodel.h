// ChromaAberrations_OFX - what the renderer asks of a lens, parametric or real
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// A lens model answers three questions for a field point (hx, hy) (normalised image position,
// desqueezed, 1 in the farthest corner) and a render wavelength l:
//  - the wavefront and the pupil, from which the PSF is computed (centred on the chief ray);
//  - where the chief ray lands, relative to the ideal (distortion-free) image point;
//  - the relative illumination.
// The parametric model writes the wavefront with Seidel / Zernike terms; the real lens model
// (src/lens) traces rays through an optical prescription.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "aperture.h"
#include "psfbank.h"
#include "wavefront.h"

namespace ca {

class LensModel
{
public:
    virtual ~LensModel() {}

    // Called once before the queries, with the image and the render wavelengths (nm).
    virtual bool prepare(const ImageGeometry& g, const std::vector<double>& nm,
                         std::function<void(int, int)> progress, std::string* error) = 0;

    // Wavefront and pupil at a field point (computePsf is then called with these, hx, hy).
    virtual void psfInputs(double hx, double hy, int l, WavefrontParams& w, ApertureParams& a) const = 0;

    // Chief ray landing minus ideal image point, in micrometres on the (desqueezed) sensor.
    virtual void displacementUm(double hx, double hy, int l, double& dx, double& dy) const = 0;

    // Relative illumination at a field point (1 on axis). 'natural' tells whether it already
    // includes the natural falloff (real lenses), or only the mechanical vignetting.
    virtual double illumination(double hx, double hy) const = 0;
    virtual bool includesNaturalFalloff() const = 0;
    // true if illumination() is cheap enough to be called for every map node
    virtual bool cheapIllumination() const { return false; }
};

// The parametric lens of steps 1 to 3
class ParametricLensModel : public LensModel
{
public:
    ParametricLensModel(const WavefrontParams& w, const ApertureParams& a) : _w(w), _a(a) {}

    bool prepare(const ImageGeometry&, const std::vector<double>& nm, std::function<void(int, int)>, std::string*) override
    {
        _nm = nm;
        return true;
    }
    void psfInputs(double, double, int, WavefrontParams& w, ApertureParams& a) const override
    {
        w = _w;
        a = _a;
    }
    void displacementUm(double hx, double hy, int l, double& dx, double& dy) const override
    {
        imageDisplacementUm(_w, hx, hy, _nm[l], dx, dy);
    }
    double illumination(double hx, double hy) const override
    {
        ApertureParams ap = _a;
        ap.fieldX = hx;
        ap.fieldY = hy;
        return apertureTransmission(ap, 128);
    }
    bool includesNaturalFalloff() const override { return false; }
    bool hasMechanicalVignetting() const { return _a.vignetting > 0.; }

private:
    WavefrontParams _w;
    ApertureParams _a;
    std::vector<double> _nm;
};

} // namespace ca
