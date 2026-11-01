// ChromaAberrations_OFX - spectral helpers
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "spectrum.h"

#include <cmath>

namespace ca {

std::vector<WavelengthSample> sampleWavelengths(int n, double loNm, double hiNm)
{
    std::vector<WavelengthSample> s;
    if (n < 1) {
        n = 1;
    }
    const double w = (hiNm - loNm) / n;
    for (int i = 0; i < n; ++i) {
        s.push_back( { loNm + (i + 0.5) * w, w } );
    }
    return s;
}

static double lobe(double nm, double mu, double s1, double s2)
{
    const double t = (nm - mu) / (nm < mu ? s1 : s2);
    return std::exp(-0.5 * t * t);
}

void cieXYZ(double nm, double& x, double& y, double& z)
{
    x = 1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2);
    y = 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1);
    z = 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8);
}

double indexBK7(double um)
{
    const double l2 = um * um;
    const double n2 = 1.0 + 1.03961212 * l2 / (l2 - 0.00600069867)
                          + 0.231792344 * l2 / (l2 - 0.0200179144)
                          + 1.01046945 * l2 / (l2 - 103.560653);
    return std::sqrt(n2);
}

double dispersionBK7(double nm)
{
    const double nd = indexBK7(0.5875618), nF = indexBK7(0.4861327), nC = indexBK7(0.6562725);
    return (indexBK7(nm * 1e-3) - nd) / (nF - nC);
}

void xyzToLinearSRGB(double X, double Y, double Z, double& r, double& g, double& b)
{
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
}

XYZWeights whiteBalancedXYZWeights(const std::vector<WavelengthSample>& samples)
{
    XYZWeights w;
    double sx = 0., sy = 0., sz = 0.;
    for (const WavelengthSample& s : samples) {
        double x, y, z;
        cieXYZ(s.nm, x, y, z);
        w.x.push_back(x * s.weight);
        w.y.push_back(y * s.weight);
        w.z.push_back(z * s.weight);
        sx += x * s.weight;
        sy += y * s.weight;
        sz += z * s.weight;
    }
    // equal-energy spectrum -> D65 white point (0.95047, 1, 1.08883)
    for (size_t i = 0; i < samples.size(); ++i) {
        w.x[i] *= 0.95047 / sx;
        w.y[i] *= 1.0 / sy;
        w.z[i] *= 1.08883 / sz;
    }
    return w;
}

} // namespace ca
