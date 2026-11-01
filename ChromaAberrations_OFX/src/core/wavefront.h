// ChromaAberrations_OFX - parametric wavefront (Seidel + Zernike)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace ca {

// Wavefront aberration of a rotationally symmetric lens, written with the field
// dependence of Seidel's theory (vector form, so any field direction works):
//
//   W = W040 (r.r)^2 + W131 (H.r)(r.r) + W222 (H.r)^2 + W220 (H.H)(r.r) + W020(lambda) (r.r)
//     + W311 (H.H)(H.r) + W111(lambda) (H.r)                      <- tilts: image displacement
//     + sum_j Z_j(r)                                              <- extra Zernike terms
//
// r = normalised pupil point (|r| <= 1), H = normalised field point (|H| = 1 at the image
// corner). All coefficients are in waves at the reference wavelength lambda0: the optical
// path difference in micrometres is coefficient * lambda0, so shorter wavelengths see more
// waves of aberration, as in a real lens.
struct WavefrontParams
{
    double fNumber = 2.8;       // working f-number
    double lambda0Um = 0.55;    // reference wavelength (micrometres)

    double W040 = 0.;           // spherical aberration
    double W131 = 0.;           // coma
    double W222 = 0.;           // astigmatism
    double W220 = 0.;           // field curvature
    double W311 = 0.;           // distortion (> 0: pincushion, < 0: barrel)
    double W020 = 0.;           // defocus (focus offset)

    double axialCA = 0.;        // defocus at 400 nm is about 1.7 x this (N-BK7 dispersion law)
    double lateralCA = 0.;      // lateral colour: magnification change with wavelength
    double spherochromatism = 0.; // relative change of W040 per unit of dispersion

    std::vector<std::pair<int, double> > zernike; // (Noll index j >= 4, RMS coefficient in waves)

    // Sampled wavefront (real lenses): opdN x opdN values in micrometres over the pupil square
    // [-1, 1]^2 (row-major, y up), interpolated bicubically and added to the terms above.
    std::shared_ptr<const std::vector<float> > opdGrid;
    int opdN = 0;
};

// Optical path difference (micrometres) at pupil point (px, py), field point (hx, hy),
// wavelength lambdaNm. The tilt terms (W311, W111) are NOT included: they only move the
// image and are returned by imageDisplacementUm().
double wavefrontOpdUm(const WavefrontParams& w, double px, double py, double hx, double hy, double lambdaNm);

// Gradient of wavefrontOpdUm with respect to the pupil coordinates (micrometres per unit of r).
void wavefrontGradientUm(const WavefrontParams& w, double px, double py, double hx, double hy, double lambdaNm,
                         double& gx, double& gy);

// Displacement of the chief ray on the sensor (micrometres) caused by distortion and
// lateral colour: -2 N grad(tilt terms).
void imageDisplacementUm(const WavefrontParams& w, double hx, double hy, double lambdaNm, double& dx, double& dy);

// Zernike polynomial, Noll ordering and normalisation (RMS 1 over the unit disk).
double zernikeNoll(int j, double px, double py);

// Noll index j -> radial order n and signed azimuthal frequency m (m > 0: cos, m < 0: sin)
void zernikeNollNM(int j, int& n, int& m);

} // namespace ca
