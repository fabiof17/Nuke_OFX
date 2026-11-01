// ChromaAberrations_OFX - parametric wavefront (Seidel + Zernike)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "wavefront.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "spectrum.h"

namespace ca {

static void nollToNM(int j, int& n, int& m)
{
    // Noll (1976): j = 1, 2, ... -> (n, m) with m signed (cos for even j, sin for odd j when m != 0)
    n = 0;
    while ( (n + 1) * (n + 2) / 2 < j ) {
        ++n;
    }
    const int k = j - n * (n + 1) / 2 - 1; // 0-based position in row n
    // the |m| values of row n in Noll order: n%2, n%2, n%2+2, n%2+2, ... (m = 0 appears once)
    int am = (n % 2) + 2 * ( (k + ( (n % 2) == 0 ? 1 : 0 ) ) / 2 );
    if (am > n) {
        am = n;
    }
    m = (am == 0) ? 0 : ( (j % 2 == 0) ? am : -am );
}

static double radial(int n, int am, double r)
{
    double s = 0.;
    for (int k = 0; k <= (n - am) / 2; ++k) {
        double c = ( (k % 2) ? -1. : 1. );
        double f = 1.;
        // (n-k)! / (k! ((n+m)/2-k)! ((n-m)/2-k)!)
        for (int i = 2; i <= n - k; ++i) f *= i;
        for (int i = 2; i <= k; ++i) f /= i;
        for (int i = 2; i <= (n + am) / 2 - k; ++i) f /= i;
        for (int i = 2; i <= (n - am) / 2 - k; ++i) f /= i;
        s += c * f * std::pow(r, n - 2 * k);
    }
    return s;
}

void zernikeNollNM(int j, int& n, int& m)
{
    nollToNM(j, n, m);
}

// bicubic (Catmull-Rom) sample of an n x n grid over [-1, 1]^2
static double sampleGrid(const std::vector<float>& g, int n, double x, double y)
{
    const double u = (x + 1.) * 0.5 * (n - 1), v = (y + 1.) * 0.5 * (n - 1);
    const int i = (int)std::floor(u), j = (int)std::floor(v);
    const double fu = u - i, fv = v - j;
    auto w = [](double t, double* c) {
        const double t2 = t * t, t3 = t2 * t;
        c[0] = -0.5 * t3 + t2 - 0.5 * t;
        c[1] = 1.5 * t3 - 2.5 * t2 + 1.;
        c[2] = -1.5 * t3 + 2. * t2 + 0.5 * t;
        c[3] = 0.5 * t3 - 0.5 * t2;
    };
    double wx[4], wy[4];
    w(fu, wx);
    w(fv, wy);
    double s = 0.;
    for (int b = 0; b < 4; ++b) {
        const int jj = std::min( std::max(j - 1 + b, 0), n - 1 );
        double r = 0.;
        for (int a = 0; a < 4; ++a) {
            const int ii = std::min( std::max(i - 1 + a, 0), n - 1 );
            r += wx[a] * g[(size_t)jj * n + ii];
        }
        s += wy[b] * r;
    }
    return s;
}

double zernikeNoll(int j, double px, double py)
{
    if (j < 1) {
        return 0.;
    }
    int n, m;
    nollToNM(j, n, m);
    const double r = std::sqrt(px * px + py * py);
    const double t = std::atan2(py, px);
    const int am = std::abs(m);
    const double R = radial(n, am, r);
    if (m == 0) {
        return std::sqrt(n + 1.) * R;
    }
    const double norm = std::sqrt( 2. * (n + 1.) );
    return norm * R * ( m > 0 ? std::cos(am * t) : std::sin(am * t) );
}

double wavefrontOpdUm(const WavefrontParams& w, double px, double py, double hx, double hy, double lambdaNm)
{
    const double d = dispersionBK7(lambdaNm);
    const double rr = px * px + py * py;
    const double hr = hx * px + hy * py;
    const double hh = hx * hx + hy * hy;
    double waves = w.W040 * (1. + w.spherochromatism * d) * rr * rr
                 + w.W131 * hr * rr
                 + w.W222 * hr * hr
                 + w.W220 * hh * rr
                 + (w.W020 + w.axialCA * d) * rr;
    for (size_t i = 0; i < w.zernike.size(); ++i) {
        waves += w.zernike[i].second * zernikeNoll(w.zernike[i].first, px, py);
    }
    double um = waves * w.lambda0Um;
    if (w.opdGrid && (w.opdN > 1)) {
        um += sampleGrid(*w.opdGrid, w.opdN, px, py);
    }
    return um;
}

void wavefrontGradientUm(const WavefrontParams& w, double px, double py, double hx, double hy, double lambdaNm,
                         double& gx, double& gy)
{
    const double e = 1e-5;
    gx = ( wavefrontOpdUm(w, px + e, py, hx, hy, lambdaNm) - wavefrontOpdUm(w, px - e, py, hx, hy, lambdaNm) ) / (2. * e);
    gy = ( wavefrontOpdUm(w, px, py + e, hx, hy, lambdaNm) - wavefrontOpdUm(w, px, py - e, hx, hy, lambdaNm) ) / (2. * e);
}

void imageDisplacementUm(const WavefrontParams& w, double hx, double hy, double lambdaNm, double& dx, double& dy)
{
    // tilt part: (W311 (H.H) + W111) (H.r)  ->  gradient wrt r = (W311 H.H + W111) H
    const double hh = hx * hx + hy * hy;
    const double c = (w.W311 * hh + w.lateralCA * dispersionBK7(lambdaNm)) * w.lambda0Um;
    // transverse ray aberration = -(1 / NA) dW/dr = -2 N dW/dr
    dx = -2. * w.fNumber * c * hx;
    dy = -2. * w.fNumber * c * hy;
}

} // namespace ca
