// ChromaAberrations_OFX - spectral helpers
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <vector>

namespace ca {

struct WavelengthSample
{
    double nm;      // wavelength in nanometres
    double weight;  // integration weight (band width in nm)
};

// n samples at the centres of n equal bands of [loNm, hiNm]
std::vector<WavelengthSample> sampleWavelengths(int n, double loNm = 400., double hiNm = 700.);

// CIE 1931 2-degree colour matching functions, multi-lobe Gaussian fit of
// Wyman, Sloan and Shirley (JCGT 2013). Error well under 1 % of the peak.
void cieXYZ(double nm, double& x, double& y, double& z);

// Refractive index of Schott N-BK7 (Sellmeier formula, lambda in micrometres).
double indexBK7(double um);

// Normalised dispersion of N-BK7: (n(lambda) - n_d) / (n_F - n_C).
// 0 at the d line (587.6 nm), about +1.7 at 400 nm, about -0.46 at 700 nm.
// Used to give the chromatic aberrations a realistic wavelength dependence.
double dispersionBK7(double nm);

// Linear sRGB (D65) from XYZ.
void xyzToLinearSRGB(double X, double Y, double Z, double& r, double& g, double& b);

// Weights that turn a set of spectral samples into XYZ such that an equal-energy
// spectrum gives the D65 white (Y = 1): XYZ = sum_i w_i * (wx_i, wy_i, wz_i) * value_i.
struct XYZWeights
{
    std::vector<double> x, y, z;
};
XYZWeights whiteBalancedXYZWeights(const std::vector<WavelengthSample>& samples);

} // namespace ca
