// ChromaAberrations_OFX - validation tests of the PSF engine (step 1)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Each test compares the engine to an independent reference: analytic diffraction theory,
// analytic geometric optics, or the other computation method (wave optics vs rays).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "psfbank.h"

using namespace ca;

static int gFail = 0, gChecks = 0;
#define CHECK(cond, ...) do { ++gChecks; bool ok_ = (cond); std::printf("  [%s] ", ok_ ? " ok " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); if (!ok_) ++gFail; } while (0)

static const double kPi = 3.14159265358979323846;

// energy inside a circle of radius R (pixels) around the kernel centre (pixel-centre test)
// energy inside a circle of radius R (pixels) around the kernel centre; each pixel is split
// in 8 x 8 sub-pixels so the circle edge is not quantised to whole pixels
static double energyWithin(const PsfResult& r, double R)
{
    double e = 0.;
    const double c = 0.5 * (r.size - 1);
    for (int y = 0; y < r.size; ++y) {
        for (int x = 0; x < r.size; ++x) {
            const double d = std::hypot(x - c, y - c);
            if (d > R + 1.) continue;
            double f = 1.;
            if (d > R - 1.) {
                int in = 0;
                for (int j = 0; j < 8; ++j) for (int i = 0; i < 8; ++i) {
                    in += std::hypot(x - c - 0.5 + (i + 0.5) / 8., y - c - 0.5 + (j + 0.5) / 8.) <= R;
                }
                f = in / 64.;
            }
            e += f * r.kernel[(size_t)y * r.size + x];
        }
    }
    return e;
}

static PsfResult psf(const WavefrontParams& w, const ApertureParams& a, double pitch, int S,
                     double hx, double hy, double nm, PsfMethod m)
{
    SensorParams s;
    s.pixelPitchUm = pitch;
    s.kernelSize = S;
    PsfOptions o;
    o.method = m;
    o.maxFFT = 4096;
    return computePsf(w, a, s, hx, hy, nm, o);
}

// ---------------------------------------------------------------------------
static void testAiry()
{
    std::printf("Airy pattern (perfect lens, round pupil, f/4, 550 nm)\n");
    WavefrontParams w;
    w.fNumber = 4.;
    ApertureParams a;
    const double pitch = 0.1, lN = 0.55 * 4.;
    PsfResult r = psf(w, a, pitch, 161, 0., 0., 550., eMethodWave);
    const int c = r.size / 2;
    const double peak = r.kernel[(size_t)c * r.size + c];
    double maxErr = 0.;
    for (int i = 0; i <= 70; ++i) {
        const double rr = i * pitch;
        const double x = kPi * rr / lN;
        const double th = (x < 1e-9) ? 1. : std::pow(2. * std::cyl_bessel_j(1., x) / x, 2.);
        const double v = r.kernel[(size_t)c * r.size + c + i] / peak;
        maxErr = std::max( maxErr, std::fabs(v - th) );
    }
    CHECK(maxErr < 0.01, "profile matches [2 J1(x)/x]^2: max error %.4f of the peak", maxErr);
    // first zero: parabola through the 3 samples around the minimum
    int im = 20;
    for (int i = 20; i <= 35; ++i) {
        if (r.kernel[(size_t)c * r.size + c + i] < r.kernel[(size_t)c * r.size + c + im]) {
            im = i;
        }
    }
    const double y0 = r.kernel[(size_t)c * r.size + c + im - 1], y1 = r.kernel[(size_t)c * r.size + c + im], y2 = r.kernel[(size_t)c * r.size + c + im + 1];
    const double zero = (im + 0.5 * (y0 - y2) / (y0 - 2. * y1 + y2)) * pitch;
    const double expect = 1.2197 * lN;
    CHECK(std::fabs(zero / expect - 1.) < 0.01, "first dark ring at %.4f um (theory 1.22 lambda N = %.4f um)", zero, expect);
}

static void testEnergy()
{
    std::printf("Energy conservation\n");
    WavefrontParams w;
    w.fNumber = 4.;
    ApertureParams a;
    const double lN = 0.55 * 4., pitch = 0.1;
    PsfResult r = psf(w, a, pitch, 161, 0., 0., 550., eMethodWave);
    // The wave PSF is computed over a field of lambda*N*K (K pupil samples); the far Airy tail beyond it
    // (about 1 % of the light at K = 64) is not represented, so encircled energy is compared as a ratio
    // to the energy inside 8 um, which is how the kernel is used anyway.
    const double Rref = 8.;
    auto th = [&](double R) { const double x = kPi * R / lN; return 1. - std::pow(std::cyl_bessel_j(0., x), 2.) - std::pow(std::cyl_bessel_j(1., x), 2.); };
    const double eRef = energyWithin(r, Rref / pitch);
    const double radii[3] = { 1.0, 1.2197 * lN, 3.0 * lN };
    for (double R : radii) {
        const double e = energyWithin(r, R / pitch) / eRef, t = th(R) / th(Rref);
        CHECK(std::fabs(e - t) < 2e-3, "encircled energy at %.2f um: %.5f (theory (1 - J0^2 - J1^2), same normalisation: %.5f)", R, e, t);
    }
    // an aberrated spot that fits in the kernel keeps all its light
    WavefrontParams wd;
    wd.fNumber = 4.;
    wd.W020 = 10.;
    PsfResult h = psf(wd, a, 2., 121, 0., 0., 550., eMethodGeometric);
    CHECK(std::fabs(h.energyInKernel - 1.) < 2e-3, "hybrid PSF, 10 waves of defocus: energy %.5f", h.energyInKernel);
    PsfResult v = psf(wd, a, 2., 121, 0., 0., 550., eMethodWave);
    CHECK(std::fabs(v.energyInKernel - 1.) < 5e-3, "wave PSF, 10 waves of defocus: energy %.5f (diffraction tails outside the kernel)", v.energyInKernel);
}

static void testSeidel()
{
    std::printf("Seidel aberrations against geometric optics (f/4, 550 nm, 2 um pixels)\n");
    ApertureParams a;
    const double N = 4., pitch = 2.;
    {
        WavefrontParams w;
        w.fNumber = N;
        w.W040 = 10.;   // 5.5 um
        const double c = 8. * N * 5.5 / pitch;         // marginal ray blur radius, pixels
        const double ee80 = c * std::pow(0.8, 1.5);    // EE(r) = (r/c)^(2/3)
        PsfResult g = psf(w, a, pitch, 201, 0., 0., 550., eMethodGeometric);
        PsfResult v = psf(w, a, pitch, 201, 0., 0., 550., eMethodWave);
        CHECK(std::fabs(g.ee80Px / ee80 - 1.) < 0.02, "spherical 10 waves, rays: EE80 %.2f px (theory %.2f)", g.ee80Px, ee80);
        CHECK(std::fabs(v.ee80Px / ee80 - 1.) < 0.05, "spherical 10 waves, wave optics: EE80 %.2f px (theory %.2f)", v.ee80Px, ee80);
    }
    {
        WavefrontParams w;
        w.fNumber = N;
        w.W020 = 10.;
        const double R = 4. * N * 5.5 / pitch;          // uniform disk
        const double ee80 = R * std::sqrt(0.8);
        PsfResult g = psf(w, a, pitch, 121, 0., 0., 550., eMethodGeometric);
        PsfResult v = psf(w, a, pitch, 121, 0., 0., 550., eMethodWave);
        CHECK(std::fabs(g.ee80Px / ee80 - 1.) < 0.02, "defocus 10 waves, rays: EE80 %.2f px (theory %.2f)", g.ee80Px, ee80);
        CHECK(std::fabs(v.ee80Px / ee80 - 1.) < 0.05, "defocus 10 waves, wave optics: EE80 %.2f px (theory %.2f)", v.ee80Px, ee80);
    }
    {
        WavefrontParams w;
        w.fNumber = N;
        w.W131 = 8.;
        PsfResult g = psf(w, a, pitch, 161, 0., 1., 550., eMethodGeometric);
        PsfResult v = psf(w, a, pitch, 161, 0., 1., 550., eMethodWave);
        CHECK(std::fabs(v.ee80Px / g.ee80Px - 1.) < 0.05, "coma 8 waves: EE80 wave %.2f px vs rays %.2f px", v.ee80Px, g.ee80Px);
        // geometric coma flare: centroid at -2N * 2/3 * W131 (theory: mean of 3 rho^2 ... = 2/3 of the tangential length/3)
        CHECK( (g.centroidYPx * v.centroidYPx > 0.) && std::fabs(v.centroidYPx / g.centroidYPx - 1.) < 0.05,
               "coma flare on the same side for both methods: centroid y %.2f px (wave) vs %.2f px (rays)", v.centroidYPx, g.centroidYPx);
        // W = W131 y (x^2 + y^2): dW/dy = x^2 + 3 y^2, whose mean over the unit disk is 1/4 + 3/4 = 1
        const double theory = -2. * N * 8. * 0.55 / pitch;
        CHECK(std::fabs(g.centroidYPx / theory - 1.) < 0.02, "coma centroid shift %.2f px (theory -2N W131 <x^2+3y^2> = %.2f px)", g.centroidYPx, theory);
    }
    {
        WavefrontParams w;
        w.fNumber = N;
        w.W222 = 8.;
        PsfResult g = psf(w, a, pitch, 161, 1., 0., 550., eMethodGeometric);
        PsfResult v = psf(w, a, pitch, 161, 1., 0., 550., eMethodWave);
        CHECK(std::fabs(v.ee80Px / g.ee80Px - 1.) < 0.05, "astigmatism 8 waves: EE80 wave %.2f px vs rays %.2f px", v.ee80Px, g.ee80Px);
    }
}

static int countSpikes(const PsfResult& r, double r0, double r1)
{
    const int nb = 360;
    std::vector<double> ang(nb, 0.);
    const double c = 0.5 * (r.size - 1);
    for (int y = 0; y < r.size; ++y) {
        for (int x = 0; x < r.size; ++x) {
            const double rr = std::hypot(x - c, y - c);
            if ( (rr < r0) || (rr > r1) ) {
                continue;
            }
            double t = std::atan2(y - c, x - c) * 180. / kPi;
            if (t < 0.) t += 360.;
            // r^2 flattens the radial fall-off so every radius counts
            ang[(int)t % nb] += r.kernel[(size_t)y * r.size + x] * rr * rr;
        }
    }
    // the number of spikes is the dominant angular harmonic of the profile
    int best = 0;
    double bestMag = 0.;
    for (int m = 1; m <= 40; ++m) {
        double re = 0., im = 0.;
        for (int i = 0; i < nb; ++i) {
            const double t = (i + 0.5) * 2. * kPi / nb;
            re += ang[i] * std::cos(m * t);
            im += ang[i] * std::sin(m * t);
        }
        const double mag = std::hypot(re, im);
        if (mag > bestMag) {
            bestMag = mag;
            best = m;
        }
    }
    return best;
}

static void testDiffractionStar()
{
    std::printf("Diffraction stars of a bladed diaphragm (f/8, 550 nm)\n");
    WavefrontParams w;
    w.fNumber = 8.;
    ApertureParams a;
    a.roundness = 0.;
    const int blades[3] = { 5, 6, 7 };
    for (int b : blades) {
        a.blades = b;
        PsfResult r = psf(w, a, 0.25, 257, 0., 0., 550., eMethodWave);
        const int expect = (b % 2) ? 2 * b : b;
        const int n = countSpikes(r, 40., 125.);
        CHECK(n == expect, "%d blades: %d spikes (theory %d)", b, n, expect);
    }
}

static void testZernike()
{
    std::printf("Zernike polynomials (Noll): RMS 1 and orthogonality over the unit disk\n");
    const int n = 600;
    double worstRms = 0., worstDot = 0.;
    for (int j = 4; j <= 22; ++j) {
        for (int k = j; k <= 22; ++k) {
            double s = 0.;
            int cnt = 0;
            for (int yy = 0; yy < n; ++yy) {
                for (int xx = 0; xx < n; ++xx) {
                    const double x = -1. + (xx + 0.5) * 2. / n, y = -1. + (yy + 0.5) * 2. / n;
                    if (x * x + y * y > 1.) continue;
                    s += zernikeNoll(j, x, y) * zernikeNoll(k, x, y);
                    ++cnt;
                }
            }
            s /= cnt;
            if (j == k) worstRms = std::max(worstRms, std::fabs(std::sqrt(s) - 1.));
            else worstDot = std::max(worstDot, std::fabs(s));
        }
    }
    CHECK(worstRms < 0.01, "j = 4..22: RMS within %.4f of 1", worstRms);
    CHECK(worstDot < 0.01, "j = 4..22: largest cross product %.4f", worstDot);
}

static void testDisplacement()
{
    std::printf("Distortion and lateral colour (chief ray displacement)\n");
    ApertureParams a;
    WavefrontParams w;
    w.fNumber = 4.;
    w.W311 = 2.;
    w.lateralCA = 1.;
    const double pitch = 5.;
    PsfResult r550 = psf(w, a, pitch, 33, 0.6, 0.8, 587.5618, eMethodAuto);   // d line: no lateral colour
    const double expectX = -2. * 4. * 2. * 0.55 * 1.0 * 0.6 / pitch, expectY = -2. * 4. * 2. * 0.55 * 1.0 * 0.8 / pitch;
    CHECK(std::fabs(r550.dispXPx - expectX) < 1e-3 && std::fabs(r550.dispYPx - expectY) < 1e-3,
          "distortion at |H| = 1: (%.4f, %.4f) px (theory %.4f, %.4f)", r550.dispXPx, r550.dispYPx, expectX, expectY);
    PsfResult r400 = psf(w, a, pitch, 33, 0.6, 0.8, 400., eMethodAuto);
    PsfResult r700 = psf(w, a, pitch, 33, 0.6, 0.8, 700., eMethodAuto);
    const double dd = (dispersionBK7(400.) - dispersionBK7(700.)) * -2. * 4. * 0.55 * 0.8 / pitch;
    CHECK(std::fabs( (r400.dispYPx - r700.dispYPx) - dd ) < 1e-3, "lateral colour 400 vs 700 nm: %.4f px (theory %.4f)", r400.dispYPx - r700.dispYPx, dd);
    // a perfect lens gives a centred PSF anywhere in the field
    WavefrontParams p0;
    p0.fNumber = 4.;
    PsfResult c = psf(p0, a, 1., 65, 0.7, -0.7, 550., eMethodWave);
    CHECK(std::hypot(c.centroidXPx, c.centroidYPx) < 0.01, "perfect lens off axis: centroid (%.4f, %.4f) px", c.centroidXPx, c.centroidYPx);
}

static void testBankTiming()
{
    std::printf("PSF bank timing (7 x 5 grid, 8 wavelengths, kernel 65, 2K image)\n");
    WavefrontParams w;
    w.fNumber = 2.8;
    w.W040 = 1.5; w.W131 = 2.; w.W222 = 1.5; w.W220 = 1.; w.axialCA = 1.;
    ApertureParams a;
    a.blades = 7;
    a.roundness = 0.3;
    ImageGeometry g;
    PsfOptions o;
    const auto t0 = std::chrono::steady_clock::now();
    PsfBank b = computePsfBank(w, a, g, 65, 7, 5, 8, o);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int wave = 0;
    for (const PsfResult& r : b.psf) wave += r.method == "wave";
    CHECK(b.psf.size() == 280, "280 PSFs in %.2f s (%d wave, %d hybrid)", s, wave, (int)b.psf.size() - wave);
}

int main()
{
    testAiry();
    testEnergy();
    testSeidel();
    testDiffractionStar();
    testZernike();
    testDisplacement();
    testBankTiming();
    std::printf("\n%d checks, %d failures\n", gChecks, gFail);
    return gFail ? 1 : 0;
}
