// ChromaAberrations_OFX - bank of PSFs over a field grid and a set of wavelengths
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "psfbank.h"
#include "lensmodel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

namespace ca {

void pixelToField(const ImageGeometry& g, double px, double py, double& hx, double& hy)
{
    // work in sensor units (pixel aspect applied to x), normalised by the distance from the
    // optical centre to the farthest corner
    const double cx = g.centerX * g.width, cy = g.centerY * g.height;
    const double ax = g.pixelAspect;
    double rmax = 0.;
    const double corners[4][2] = { { 0., 0. }, { (double)g.width, 0. }, { 0., (double)g.height }, { (double)g.width, (double)g.height } };
    for (int i = 0; i < 4; ++i) {
        rmax = std::max( rmax, std::hypot( (corners[i][0] - cx) * ax, corners[i][1] - cy ) );
    }
    hx = (px - cx) * ax / rmax;
    hy = (py - cy) / rmax;
}

// Area-weighted 1D resampling matrix: source bins of width q (n of them, centred on 0) into
// target bins of width t (m of them, centred on 0). w[target * n + source].
static std::vector<double> binMatrix(int n, double q, int m, double t)
{
    std::vector<double> w( (size_t)m * n, 0. );
    for (int a = 0; a < m; ++a) {
        const double t0 = (a - 0.5 * m) * t, t1 = t0 + t;
        for (int b = 0; b < n; ++b) {
            const double s0 = (b - 0.5 * n) * q, s1 = s0 + q;
            const double o = std::min(t1, s1) - std::max(t0, s0);
            if (o > 0.) {
                w[(size_t)a * n + b] = o / q;
            }
        }
    }
    return w;
}

// PSF computed on square bins of size q (sq x sq) -> plate pixels px x py (s x s)
static void resampleKernel(PsfResult& r, double q, double px, double py, int s)
{
    const int n = r.size;
    const std::vector<double> wx = binMatrix(n, q, s, px), wy = binMatrix(n, q, s, py);
    std::vector<double> tmp( (size_t)n * s, 0. );   // n rows, s columns
    for (int y = 0; y < n; ++y) {
        for (int a = 0; a < s; ++a) {
            double v = 0.;
            for (int x = 0; x < n; ++x) {
                v += wx[(size_t)a * n + x] * r.kernel[(size_t)y * n + x];
            }
            tmp[(size_t)y * s + a] = v;
        }
    }
    std::vector<float> k( (size_t)s * s, 0.f );
    double sum = 0.;
    for (int b = 0; b < s; ++b) {
        for (int a = 0; a < s; ++a) {
            double v = 0.;
            for (int y = 0; y < n; ++y) {
                v += wy[(size_t)b * n + y] * tmp[(size_t)y * s + a];
            }
            k[(size_t)b * s + a] = (float)v;
            sum += v;
        }
    }
    r.kernel.swap(k);
    r.size = s;
    r.energyInKernel = sum;
    double sx = 0., sy = 0.;
    const double c = 0.5 * (s - 1);
    for (int b = 0; b < s; ++b) {
        for (int a = 0; a < s; ++a) {
            sx += r.kernel[(size_t)b * s + a] * (a - c);
            sy += r.kernel[(size_t)b * s + a] * (b - c);
        }
    }
    r.centroidXPx = sum > 0. ? sx / sum : 0.;
    r.centroidYPx = sum > 0. ? sy / sum : 0.;
    r.ee80Px = encircledEnergyRadius(r.kernel, s, 0.8);
}

PsfBank computePsfBank(const WavefrontParams& w, const ApertureParams& a, const ImageGeometry& g,
                       int kernelSize, int gridX, int gridY, int nWavelengths, const PsfOptions& opt,
                       std::function<void(int, int)> progress)
{
    ParametricLensModel model(w, a);
    const std::vector<WavelengthSample> wl = sampleWavelengths(nWavelengths);
    std::vector<double> nm;
    for (const WavelengthSample& s : wl) {
        nm.push_back(s.nm);
    }
    model.prepare(g, nm, std::function<void(int, int)>(), 0);
    return computePsfBank(model, g, kernelSize, gridX, gridY, wl, opt, progress);
}

PsfResult computePsfAt(const LensModel& model, const ImageGeometry& g, int kernelSize, double px, double py,
                       int l, double nm, const PsfOptions& opt)
{
    // Plate pixels are pixelPitch tall and pixelPitch x pixelAspect wide in the desqueezed
    // (object) space where the lens is modelled. With non-square pixels the PSF is computed on
    // square bins of the smaller size, then binned onto the plate pixels.
    const double ax = g.pixelAspect > 0. ? g.pixelAspect : 1.;
    const bool anamorphic = std::fabs(ax - 1.) > 1e-6;
    const int S = kernelSize | 1;
    const double pitch = g.pixelPitchUm();
    const double q = pitch * std::min(1., ax);
    SensorParams s;
    s.pixelPitchUm = anamorphic ? q : pitch;
    s.kernelSize = anamorphic ? ( (int)std::ceil(S * pitch * std::max(1., ax) / q) | 1 ) : S;
    double hx, hy;
    pixelToField(g, px, py, hx, hy);
    WavefrontParams w;
    ApertureParams a;
    model.psfInputs(hx, hy, l, w, a);
    PsfResult r = computePsf(w, a, s, hx, hy, nm, opt);
    if (anamorphic) {
        resampleKernel(r, q, pitch * ax, pitch, S);
    }
    // chief ray displacement from the model, in plate pixels
    double dx, dy;
    model.displacementUm(hx, hy, l, dx, dy);
    r.dispXPx = dx / (pitch * ax);
    r.dispYPx = dy / pitch;
    return r;
}

double psfExtentPx(const LensModel& model, const ImageGeometry& g, double px, double py, int l, double nm)
{
    double hx, hy;
    pixelToField(g, px, py, hx, hy);
    WavefrontParams w;
    ApertureParams a;
    model.psfInputs(hx, hy, l, w, a);
    // geometric spot (largest transverse aberration) + diffraction (rings, and the spikes of a
    // bladed diaphragm, which show on bright highlights)
    const double lN = nm * 1e-3 * w.fNumber;
    const bool spikes = (a.blades >= 3) && (a.roundness < 1.);
    const double um = psfGeometricRadiusUm(w, a, hx, hy, nm) + (spikes ? 24. : 8.) * lN;
    const double ax = g.pixelAspect > 0. ? g.pixelAspect : 1.;
    return um / ( g.pixelPitchUm() * std::min(1., ax) );
}

PsfBank computePsfBank(const LensModel& model, const ImageGeometry& g, int kernelSize, int gridX, int gridY,
                       const std::vector<WavelengthSample>& wavelengths, const PsfOptions& opt,
                       std::function<void(int, int)> progress)
{
    PsfBank b;
    b.gridX = std::max(1, gridX);
    b.gridY = std::max(1, gridY);
    b.wavelengths = wavelengths;
    const int nl = (int)b.wavelengths.size();
    const int cells = b.gridX * b.gridY;
    b.psf.resize( (size_t)cells * nl );
    for (int gy = 0; gy < b.gridY; ++gy) {
        for (int gx = 0; gx < b.gridX; ++gx) {
            // cells centred on a regular grid that reaches the image borders
            const double px = b.gridX > 1 ? g.width * gx / (b.gridX - 1.) : 0.5 * g.width;
            const double py = b.gridY > 1 ? g.height * gy / (b.gridY - 1.) : 0.5 * g.height;
            double hx, hy;
            pixelToField(g, px, py, hx, hy);
            b.pixelX.push_back(px);
            b.pixelY.push_back(py);
            b.fieldX.push_back(hx);
            b.fieldY.push_back(hy);
        }
    }
    const int total = cells * nl;
    std::atomic<int> next(0), done(0);
    std::mutex progressMutex;
    auto worker = [&]() {
        for (;;) {
            const int t = next++;
            if (t >= total) {
                return;
            }
            const int cell = t / nl, l = t % nl;
            b.psf[t] = computePsfAt(model, g, kernelSize, b.pixelX[cell], b.pixelY[cell], l, b.wavelengths[l].nm, opt);
            const int d = ++done;
            if (progress) {
                std::lock_guard<std::mutex> lock(progressMutex);
                progress(d, total);
            }
        }
    };
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    for (unsigned i = 1; i < nt; ++i) {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread& t : threads) {
        t.join();
    }
    return b;
}

} // namespace ca
