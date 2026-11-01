// ChromaAberrations_OFX - point spread function engine
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Wave method: the pupil function A(r) exp(-i 2 pi W(r) / lambda) is sampled on a K x K grid
// across the pupil diameter, zero-padded to M = K Q, Fourier transformed, squared and
// integrated over the sensor pixels. Sample spacing in the image: lambda N / Q.
//
// Hybrid method (large aberrations, where the wave method would need a huge FFT): the
// geometric spot (pupil samples sent to -2 N grad W) is convolved with the diffraction PSF
// of the unaberrated aperture.

#include "psf.h"

#include <algorithm>
#include <cmath>
#include <complex>

#include "pocketfft_hdronly.h"

namespace ca {

static const double kPi = 3.14159265358979323846;

typedef std::complex<float> cfloat;

static void fft2d(std::vector<cfloat>& data, size_t ny, size_t nx, bool forward)
{
    pocketfft::shape_t shape = { ny, nx };
    pocketfft::stride_t stride = { (ptrdiff_t)(nx * sizeof(cfloat)), (ptrdiff_t)sizeof(cfloat) };
    pocketfft::shape_t axes = { 0, 1 };
    pocketfft::c2c(shape, stride, stride, axes, forward, data.data(), data.data(), 1.f, 1);
}

// largest |grad W| over the open pupil, micrometres per unit of r
static double maxSlopeUm(const WavefrontParams& w, const ApertureParams& a, double hx, double hy, double nm)
{
    double m = 0.;
    const int n = 48;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            const double x = -1. + 2. * i / n, y = -1. + 2. * j / n;
            if (x * x + y * y > 1.) {
                continue;
            }
            double gx, gy;
            wavefrontGradientUm(w, x, y, hx, hy, nm, gx, gy);
            m = std::max(m, std::sqrt(gx * gx + gy * gy));
        }
    }
    (void)a;
    return m;
}

// Area-weighted binning of a centred fine grid (spacing d, n samples, sample k at
// (k - n/2) d, or its FFT-wrapped index) onto S pixels of pitch p centred on 0.
struct Bin
{
    int pix[3];
    double w[3];
    int count;
};

static std::vector<Bin> makeBins(int n, double d, double p, int S, bool fftOrder)
{
    std::vector<Bin> bins(n);
    for (int k = 0; k < n; ++k) {
        const int kk = fftOrder ? (k < n / 2 ? k : k - n) : k - n / 2;
        const double u0 = kk * d - 0.5 * d, u1 = kk * d + 0.5 * d;
        Bin& b = bins[k];
        b.count = 0;
        const int c0 = (int)std::floor(u0 / p + 0.5 * S);
        const int c1 = (int)std::floor(u1 / p + 0.5 * S - 1e-12);
        for (int c = c0; c <= c1 && b.count < 3; ++c) {
            if ( (c < 0) || (c >= S) ) {
                continue;
            }
            const double lo = std::max(u0, (c - 0.5 * S) * p);
            const double hi = std::min(u1, (c + 1 - 0.5 * S) * p);
            if (hi > lo) {
                b.pix[b.count] = c;
                b.w[b.count] = (hi - lo) / d;
                ++b.count;
            }
        }
    }
    return bins;
}

// separable binning of an n x n intensity grid into an S x S kernel
static void binGrid(const std::vector<float>& I, int n, const std::vector<Bin>& bins, int S, std::vector<float>& kernel)
{
    std::vector<double> tmp( (size_t)n * S, 0. );
    for (int y = 0; y < n; ++y) {
        const float* row = &I[(size_t)y * n];
        double* t = &tmp[(size_t)y * S];
        for (int x = 0; x < n; ++x) {
            const Bin& b = bins[x];
            for (int q = 0; q < b.count; ++q) {
                t[b.pix[q]] += row[x] * b.w[q];
            }
        }
    }
    std::vector<double> k( (size_t)S * S, 0. );
    for (int y = 0; y < n; ++y) {
        const Bin& b = bins[y];
        const double* t = &tmp[(size_t)y * S];
        for (int q = 0; q < b.count; ++q) {
            double* kr = &k[(size_t)b.pix[q] * S];
            const double wq = b.w[q];
            for (int x = 0; x < S; ++x) {
                kr[x] += t[x] * wq;
            }
        }
    }
    kernel.assign( k.begin(), k.end() );
}

static bool wavePsf(const WavefrontParams& w, const ApertureParams& a, const SensorParams& s,
                    double hx, double hy, double nm, const PsfOptions& opt, bool force, PsfResult& res)
{
    const double lambdaUm = nm * 1e-3;
    const double N = w.fNumber, p = s.pixelPitchUm;
    const int S = s.kernelSize | 1;
    const double lN = lambdaUm * N;

    // Pupil sampling. The image window is lambda N K wide, so K sets both the phase sampling
    // and the window size:
    //  - phase step at most 1/3 wave per sample (K >= 6 slope);
    //  - a window of at least 128 lambda N (64 diffraction rings on each side) that holds twice
    //    the geometric spot diameter (K >= 8 slope), but no larger than the kernel itself:
    //    kernel pixels outside the window receive the (negligible) far tail as zero.
    // Without the last cap, large pixels (12 um at 2K on Super 35) against a 1 um Airy disk
    // forced 2000^2 transforms for every PSF.
    const double slopeWaves = maxSlopeUm(w, a, hx, hy, nm) / lambdaUm;
    const int kPhase = (int)std::ceil(6. * slopeWaves);
    const int kWindow = std::max( 128, (int)std::ceil(8. * slopeWaves) );
    const int kKernel = (int)std::ceil(S * p / lN * 1.15);
    int K = std::max( 64, std::max( kPhase, std::min(kWindow, kKernel) ) );
    K += K & 1;
    const int Q = std::max( 2, (int)std::ceil(2. * lN / p) );
    const int M = K * Q;
    if ( (M > opt.maxFFT) && !(force && M <= 2 * opt.maxFFT) ) {
        return false;
    }
    const double d = lN / Q;              // image sample spacing (micrometres)
    const double cell = 2. / K;           // pupil sample spacing (normalised)

    std::vector<cfloat> U( (size_t)M * M, cfloat(0.f, 0.f) );
    const int half = M / 2;
    const int r0 = std::max(0, half - K / 2 - 2), r1 = std::min(M, half + K / 2 + 2);
    for (int j = r0; j < r1; ++j) {
        const double y = (j - half) * cell;
        for (int i = r0; i < r1; ++i) {
            const double x = (i - half) * cell;
            if (x * x + y * y > (1. + cell) * (1. + cell)) {
                continue;
            }
            const double A = apertureCoverage(a, x, y, cell);
            if (A <= 0.) {
                continue;
            }
            const double phase = -2. * kPi * wavefrontOpdUm(w, x, y, hx, hy, nm) / lambdaUm;
            U[(size_t)j * M + i] = cfloat( (float)(A * std::cos(phase)), (float)(A * std::sin(phase)) );
        }
    }
    fft2d(U, M, M, true);

    std::vector<float> I( (size_t)M * M );
    double total = 0.;
    for (size_t k = 0; k < I.size(); ++k) {
        I[k] = std::norm(U[k]);
        total += I[k];
    }
    if (total <= 0.) {
        return false;
    }
    const float inv = (float)(1. / total);
    for (size_t k = 0; k < I.size(); ++k) {
        I[k] *= inv;
    }
    const std::vector<Bin> bins = makeBins(M, d, p, S, true);
    binGrid(I, M, bins, S, res.kernel);
    res.size = S;
    res.method = "wave";
    res.pupilSamples = K;
    res.fftSize = M;
    return true;
}

static unsigned int hashu(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

static void hybridPsf(const WavefrontParams& w, const ApertureParams& a, const SensorParams& s,
                      double hx, double hy, double nm, const PsfOptions& opt, PsfResult& res)
{
    const int S = s.kernelSize | 1;
    const int F = 4;                       // fine grid: 4 x 4 samples per pixel
    const int Sf = S * F;
    const double g = s.pixelPitchUm / F;
    const double N = w.fNumber;

    // 1. geometric spot
    std::vector<double> spot( (size_t)Sf * Sf, 0. );
    const int n = std::max(64, opt.geoSamples);
    long long inside = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const unsigned int h = hashu( (unsigned)(j * n + i) * 2654435761u );
            const double jx = (h & 0xffff) / 65536., jy = (h >> 16) / 65536.;
            const double x = -1. + 2. * (i + jx) / n, y = -1. + 2. * (j + jy) / n;
            if ( !apertureInside(a, x, y) ) {
                continue;
            }
            ++inside;
            double gx, gy;
            wavefrontGradientUm(w, x, y, hx, hy, nm, gx, gy);
            // fine-pixel coordinates, pixel c covering [c, c + 1)
            const double fx = -2. * N * gx / g + 0.5 * Sf - 0.5;
            const double fy = -2. * N * gy / g + 0.5 * Sf - 0.5;
            const int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
            const double tx = fx - ix, ty = fy - iy;
            for (int dy = 0; dy < 2; ++dy) {
                const int yy = iy + dy;
                if ( (yy < 0) || (yy >= Sf) ) {
                    continue;
                }
                for (int dx = 0; dx < 2; ++dx) {
                    const int xx = ix + dx;
                    if ( (xx < 0) || (xx >= Sf) ) {
                        continue;
                    }
                    spot[(size_t)yy * Sf + xx] += (dx ? tx : 1. - tx) * (dy ? ty : 1. - ty);
                }
            }
        }
    }
    if (inside > 0) {
        for (double& v : spot) {
            v /= (double)inside;
        }
    } else {
        // closed pupil (fully vignetted field): no light gets through; a point keeps the kernel
        // well defined, the relative illumination carries the darkness
        spot[(size_t)(Sf / 2) * Sf + Sf / 2] = 1.;
    }

    // 2. diffraction PSF of the unaberrated aperture on the fine grid
    WavefrontParams w0;
    w0.fNumber = w.fNumber;
    w0.lambda0Um = w.lambda0Um;
    SensorParams s0;
    s0.pixelPitchUm = g;
    int Sd = (int)std::ceil(16. * nm * 1e-3 * N / g) | 1;
    Sd = std::max(9, std::min(Sd, Sf | 1));
    s0.kernelSize = Sd;
    PsfOptions o0;
    o0.maxFFT = 1 << 14;
    PsfResult diff;
    if ( (inside == 0) || !wavePsf(w0, a, s0, 0., 0., nm, o0, true, diff) || (diff.size != Sd) ) {
        Sd = 1;
        diff.size = 1;
        diff.kernel.assign(1, 1.f);
    }
    double dsum = 0.;
    for (float v : diff.kernel) {
        dsum += v;
    }
    // the light the diffraction kernel misses (far rings) is put back by renormalising
    const double dnorm = dsum > 0. ? 1. / dsum : 1.;

    // 3. convolution by FFT (linear, "same" size)
    int P = Sf + Sd - 1;
    P += P & 1;
    std::vector<cfloat> A( (size_t)P * P, cfloat(0.f, 0.f) ), B( (size_t)P * P, cfloat(0.f, 0.f) );
    for (int y = 0; y < Sf; ++y) {
        for (int x = 0; x < Sf; ++x) {
            A[(size_t)y * P + x] = cfloat( (float)spot[(size_t)y * Sf + x], 0.f );
        }
    }
    for (int y = 0; y < Sd; ++y) {
        for (int x = 0; x < Sd; ++x) {
            B[(size_t)y * P + x] = cfloat( (float)(diff.kernel[(size_t)y * Sd + x] * dnorm), 0.f );
        }
    }
    fft2d(A, P, P, true);
    fft2d(B, P, P, true);
    for (size_t k = 0; k < A.size(); ++k) {
        A[k] *= B[k];
    }
    fft2d(A, P, P, false);
    const float invP = 1.f / ( (float)P * (float)P );
    const int off = Sd / 2;

    // 4. back to pixels (F x F box)
    res.kernel.assign( (size_t)S * S, 0.f );
    for (int y = 0; y < Sf; ++y) {
        for (int x = 0; x < Sf; ++x) {
            const float v = std::max(0.f, A[(size_t)(y + off) * P + (x + off)].real() * invP);
            res.kernel[(size_t)(y / F) * S + (x / F)] += v;
        }
    }
    res.size = S;
    res.method = "hybrid";
    res.pupilSamples = 0;
    res.fftSize = P;
}

double encircledEnergyRadius(const std::vector<float>& kernel, int size, double fraction)
{
    std::vector<std::pair<double, double> > rv;
    rv.reserve(kernel.size());
    const double c = 0.5 * (size - 1);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            rv.push_back( std::make_pair( std::hypot(x - c, y - c), (double)kernel[(size_t)y * size + x] ) );
        }
    }
    std::sort( rv.begin(), rv.end() );
    double cum = 0.;
    for (size_t i = 0; i < rv.size(); ++i) {
        const double prev = cum;
        cum += rv[i].second;
        if (cum >= fraction) {
            // interpolate between the previous radius and this one
            const double r0 = i ? rv[i - 1].first : 0.;
            const double t = rv[i].second > 0. ? (fraction - prev) / rv[i].second : 1.;
            return r0 + t * (rv[i].first - r0);
        }
    }
    return -1.;
}

double psfGeometricRadiusUm(const WavefrontParams& w, const ApertureParams& a, double hx, double hy, double nm)
{
    return 2. * w.fNumber * maxSlopeUm(w, a, hx, hy, nm);
}

PsfResult computePsf(const WavefrontParams& w, const ApertureParams& aperture, const SensorParams& s,
                     double hx, double hy, double lambdaNm, const PsfOptions& opt)
{
    PsfResult res;
    ApertureParams a = aperture;   // the pupil as seen from this field point (mechanical vignetting)
    a.fieldX = hx;
    a.fieldY = hy;
    bool done = false;
    if (opt.method != eMethodGeometric) {
        done = wavePsf(w, a, s, hx, hy, lambdaNm, opt, opt.method == eMethodWave, res);
    }
    if (!done) {
        hybridPsf(w, a, s, hx, hy, lambdaNm, opt, res);
    }
    double sum = 0., sx = 0., sy = 0.;
    const double c = 0.5 * (res.size - 1);
    for (int y = 0; y < res.size; ++y) {
        for (int x = 0; x < res.size; ++x) {
            const double v = res.kernel[(size_t)y * res.size + x];
            sum += v;
            sx += v * (x - c);
            sy += v * (y - c);
        }
    }
    res.energyInKernel = sum;
    res.centroidXPx = sum > 0. ? sx / sum : 0.;
    res.centroidYPx = sum > 0. ? sy / sum : 0.;
    double dx, dy;
    imageDisplacementUm(w, hx, hy, lambdaNm, dx, dy);
    res.dispXPx = dx / s.pixelPitchUm;
    res.dispYPx = dy / s.pixelPitchUm;
    res.ee80Px = encircledEnergyRadius(res.kernel, res.size, 0.8);
    return res;
}

} // namespace ca
