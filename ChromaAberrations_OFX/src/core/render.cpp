// ChromaAberrations_OFX - image renderer (CPU reference) and the exact renderer for the tests
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "render.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <mutex>
#include <thread>

#include "pocketfft_hdronly.h"

namespace ca {

typedef std::complex<float> cfloat;
typedef std::chrono::steady_clock Clock;

static double since(Clock::time_point t)
{
    return std::chrono::duration<double>(Clock::now() - t).count();
}

static int threadCount()
{
    return (int)std::max(1u, std::thread::hardware_concurrency());
}

// rows [0, n) split over the threads
static void parallelRows(int n, const std::function<void(int, int)>& f)
{
    const int nt = std::min(threadCount(), std::max(1, n / 16));
    if (nt <= 1) {
        f(0, n);
        return;
    }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        const int a = (int)( (long long)n * t / nt ), b = (int)( (long long)n * (t + 1) / nt );
        th.emplace_back(f, a, b);
    }
    for (std::thread& t : th) {
        t.join();
    }
}

int goodFftSize(int n)
{
    for (int m = std::max(1, n); ; ++m) {
        int r = m;
        const int f[4] = { 2, 3, 5, 7 };
        for (int p : f) {
            while (r % p == 0) {
                r /= p;
            }
        }
        if (r == 1) {
            return m;
        }
    }
}

PixelMath makePixelMath(const LensPlan& plan)
{
    PixelMath pm;
    pm.width = plan.geom.width;
    pm.height = plan.geom.height;
    pm.R = plan.R;
    pm.mapW = plan.mapW;
    pm.mapH = plan.mapH;
    pm.mapX0 = (float)plan.mapX0;
    pm.mapY0 = (float)plan.mapY0;
    pm.mapInvStep = (float)(1. / plan.mapStep);
    pm.gridX = plan.gridX;
    pm.gridY = plan.gridY;
    pm.lutN = plan.spectral.lutN;
    pm.lutLo = (float)plan.spectral.lutLo;
    pm.lutInvStep = (float)(1. / plan.spectral.lutStep);
    pm.gridSx = plan.gridX > 1 ? (float)(plan.gridX - 1) / plan.geom.width : 0.f;
    pm.gridSy = plan.gridY > 1 ? (float)(plan.gridY - 1) / plan.geom.height : 0.f;
    return pm;
}

// light of wavelength l over the padded domain (dw x dh, row stride 'stride')
static void buildSource(const LensPlan& plan, const PixelMath& pm, const ImageRGB& in, int l,
                        float* dst, int dw, int dh, int stride)
{
    const bool warp = plan.settings.distortion;
    const bool vig = plan.settings.vignetting;
    const float br = (float)plan.spectral.basis[0][l], bg = (float)plan.spectral.basis[1][l], bb = (float)plan.spectral.basis[2][l];
    const float* lut = plan.spectral.lut.empty() ? 0 : &plan.spectral.lut[(size_t)l * plan.spectral.lutN * plan.spectral.lutN];
    parallelRows(dh, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            float* row = dst + (size_t)j * stride;
            for (int i = 0; i < dw; ++i) {
                row[i] = sourceValue(pm, i, j, in.c[0].data(), in.c[1].data(), in.c[2].data(),
                                     warp ? plan.dispX[l].data() : 0, warp ? plan.dispY[l].data() : 0,
                                     vig ? plan.illum.data() : 0, lut, br, bg, bb);
            }
        }
    });
}

TileLayout makeTileLayout(const LensPlan& plan)
{
    TileLayout t;
    const int R = plan.R;
    t.dw = plan.geom.width + 2 * R;
    t.dh = plan.geom.height + 2 * R;
    int maxW = 1, maxH = 1;
    for (int gy = 0; gy < plan.gridY; ++gy) {
        for (int gx = 0; gx < plan.gridX; ++gx) {
            // padded pixel i is at image position x = i - R + 0.5; the weight of column gx is
            // not zero between the neighbouring columns
            const int i0 = gx == 0 ? 0 : std::max( 0, (int)std::floor(plan.nodeX(gx - 1) + R - 0.5) );
            const int i1 = gx == plan.gridX - 1 ? t.dw - 1 : std::min( t.dw - 1, (int)std::ceil(plan.nodeX(gx + 1) + R - 0.5) );
            const int j0 = gy == 0 ? 0 : std::max( 0, (int)std::floor(plan.nodeY(gy - 1) + R - 0.5) );
            const int j1 = gy == plan.gridY - 1 ? t.dh - 1 : std::min( t.dh - 1, (int)std::ceil(plan.nodeY(gy + 1) + R - 0.5) );
            t.i0.push_back(i0);
            t.i1.push_back(i1);
            t.j0.push_back(j0);
            t.j1.push_back(j1);
            t.ox.push_back(i0 - R);
            t.oy.push_back(j0 - R);
            maxW = std::max(maxW, i1 - i0 + 1 + 2 * R);
            maxH = std::max(maxH, j1 - j0 + 1 + 2 * R);
        }
    }
    t.tw = goodFftSize(maxW);
    t.th = goodFftSize(maxH);
    return t;
}

void renderCpu(const LensPlan& plan, const ImageRGB& in, ImageRGB& out, RenderStats* stats)
{
    const auto t0 = Clock::now();
    const int W = in.width, H = in.height;
    out.resize(W, H);
    const PixelMath pm = makePixelMath(plan);
    const int R = plan.R, S = plan.S, P = S * S;
    const int nl = plan.wavelengthCount();
    const int dw = W + 2 * R, dh = H + 2 * R;
    double tSource = 0., tFft = 0.;

    // light of every wavelength over the padded domain
    auto ts = Clock::now();
    std::vector<std::vector<float> > f(nl);
    for (int l = 0; l < nl; ++l) {
        f[l].resize( (size_t)dw * dh );
        buildSource(plan, pm, in, l, f[l].data(), dw, dh, dw);
    }
    tSource = since(ts);

    if (S == 1) {
        // no blur: accumulate the wavelengths directly
        for (int l = 0; l < nl; ++l) {
            for (int c = 0; c < 3; ++c) {
                const float wc = (float)plan.spectral.response[c][l];
                float* o = out.c[c].data();
                const float* src = f[l].data();
                for (size_t p = 0; p < f[l].size(); ++p) {
                    o[p] += wc * src[p];
                }
            }
        }
        if (stats) {
            stats->fftWidth = stats->fftHeight = 0;
            stats->convolutions = 0;
        }
    } else {
        ts = Clock::now();
        const TileLayout tl = makeTileLayout(plan);
        const int tw = tl.tw, th = tl.th, cw = tw / 2 + 1;
        const size_t nReal = (size_t)tw * th, nCplx = (size_t)cw * th;
        const pocketfft::shape_t shape = { (size_t)th, (size_t)tw };
        const pocketfft::stride_t sR = { (ptrdiff_t)(tw * sizeof(float)), (ptrdiff_t)sizeof(float) };
        const pocketfft::stride_t sC = { (ptrdiff_t)(cw * sizeof(cfloat)), (ptrdiff_t)sizeof(cfloat) };
        const pocketfft::shape_t axes = { 0, 1 };
        const float norm = 1.f / (float)nReal;
        const int G = plan.nodeCount();
        std::atomic<int> next(0);
        std::mutex outMutex;
        auto worker = [&]() {
            std::vector<float> tile(nReal), ker(nReal);
            std::vector<cfloat> T(nCplx), E(nCplx), acc[3];
            for (int c = 0; c < 3; ++c) {
                acc[c].resize(nCplx);
            }
            for (;;) {
                const int g = next++;
                if (g >= G) {
                    return;
                }
                const int gx = g % plan.gridX, gy = g / plan.gridX;
                const int i0 = tl.i0[g], i1 = tl.i1[g], j0 = tl.j0[g], j1 = tl.j1[g], ox = tl.ox[g], oy = tl.oy[g];
                for (int c = 0; c < 3; ++c) {
                    std::fill(acc[c].begin(), acc[c].end(), cfloat(0.f, 0.f));
                }
                for (int l = 0; l < nl; ++l) {
                    // weighted light of this grid point
                    std::fill(tile.begin(), tile.end(), 0.f);
                    for (int j = j0; j <= j1; ++j) {
                        const float* src = &f[l][(size_t)j * dw];
                        float* dst = &tile[(size_t)(j - oy) * tw];
                        for (int i = i0; i <= i1; ++i) {
                            dst[i - ox] = src[i] * nodeWeight(pm, gx, gy, i, j);
                        }
                    }
                    // its PSF, centred at the origin with wrap-around
                    std::fill(ker.begin(), ker.end(), 0.f);
                    const float* k = &plan.bankKernels[l][(size_t)g * P];
                    for (int y = 0; y < S; ++y) {
                        const int yy = (y - R + th) % th;
                        for (int x = 0; x < S; ++x) {
                            ker[(size_t)yy * tw + (x - R + tw) % tw] = k[(size_t)y * S + x];
                        }
                    }
                    pocketfft::r2c(shape, sR, sC, axes, true, tile.data(), T.data(), 1.f, 1);
                    pocketfft::r2c(shape, sR, sC, axes, true, ker.data(), E.data(), 1.f, 1);
                    const float w0 = (float)plan.spectral.response[0][l] * norm;
                    const float w1 = (float)plan.spectral.response[1][l] * norm;
                    const float w2 = (float)plan.spectral.response[2][l] * norm;
                    for (size_t p = 0; p < nCplx; ++p) {
                        const cfloat v = T[p] * E[p];
                        acc[0][p] += w0 * v;
                        acc[1][p] += w1 * v;
                        acc[2][p] += w2 * v;
                    }
                }
                // back to the image: tile index (t, s) is padded pixel (ox + t, oy + s)
                const int y0 = std::max(0, oy - R), y1 = std::min(H, oy + th - R);
                const int x0 = std::max(0, ox - R), x1 = std::min(W, ox + tw - R);
                for (int c = 0; c < 3; ++c) {
                    pocketfft::c2r(shape, sC, sR, axes, false, acc[c].data(), tile.data(), 1.f, 1);
                    std::lock_guard<std::mutex> lock(outMutex);
                    float* o = out.c[c].data();
                    for (int y = y0; y < y1; ++y) {
                        const float* row = &tile[(size_t)(y + R - oy) * tw];
                        float* orow = &o[(size_t)y * W];
                        for (int x = x0; x < x1; ++x) {
                            orow[x] += row[x + R - ox];
                        }
                    }
                }
            }
        };
        const int nt = std::min(threadCount(), G);
        std::vector<std::thread> th2;
        for (int t = 1; t < nt; ++t) {
            th2.emplace_back(worker);
        }
        worker();
        for (std::thread& t : th2) {
            t.join();
        }
        tFft = since(ts);
        if (stats) {
            stats->fftWidth = tw;
            stats->fftHeight = th;
            stats->convolutions = G * nl;
        }
    }
    if (stats) {
        stats->seconds = since(t0);
        stats->sourceSeconds = tSource;
        stats->fftSeconds = tFft;
        stats->device = "CPU";
    }
}

void renderExactCpu(const LensPlan& plan, const ImageRGB& in, ImageRGB& out)
{
    const int W = in.width, H = in.height;
    out.resize(W, H);
    const PixelMath pm = makePixelMath(plan);
    const int R = plan.R, S = plan.S, P = S * S;
    const int nl = plan.wavelengthCount();
    const int dw = W + 2 * R, dh = H + 2 * R;
    std::vector<float> f( (size_t)dw * dh );
    std::vector<double> acc[3];
    for (int c = 0; c < 3; ++c) {
        acc[c].assign( (size_t)W * H, 0. );
    }
    for (int l = 0; l < nl; ++l) {
        buildSource(plan, pm, in, l, f.data(), dw, dh, dw);
        const std::vector<float>& bank = plan.bankKernels[l];
        // threads own bands of output rows; each source row is visited by the bands it reaches
        const int nt = threadCount();
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t) {
            th.emplace_back([&, t]() {
                const int oy0 = (int)( (long long)H * t / nt ), oy1 = (int)( (long long)H * (t + 1) / nt );
                std::vector<float> psf(P);
                for (int j = std::max(0, oy0); j < std::min(dh, oy1 + 2 * R); ++j) {
                    for (int i = 0; i < dw; ++i) {
                        const float v = f[(size_t)j * dw + i];
                        if (v == 0.f) {
                            continue;
                        }
                        // PSF at the landing position: bilinear between the 4 grid PSFs
                        const float x = i - R + 0.5f, y = j - R + 0.5f;
                        const float u = caClamp(x * pm.gridSx, 0.f, (float)(pm.gridX - 1));
                        const float vv = caClamp(y * pm.gridSy, 0.f, (float)(pm.gridY - 1));
                        int gi = (int)u, gj = (int)vv;
                        if (gi > pm.gridX - 2) gi = std::max(0, pm.gridX - 2);
                        if (gj > pm.gridY - 2) gj = std::max(0, pm.gridY - 2);
                        const float fu = u - gi, fv = vv - gj;
                        const int gi1 = std::min(gi + 1, pm.gridX - 1), gj1 = std::min(gj + 1, pm.gridY - 1);
                        const float* k00 = &bank[(size_t)(gj * pm.gridX + gi) * P];
                        const float* k01 = &bank[(size_t)(gj * pm.gridX + gi1) * P];
                        const float* k10 = &bank[(size_t)(gj1 * pm.gridX + gi) * P];
                        const float* k11 = &bank[(size_t)(gj1 * pm.gridX + gi1) * P];
                        const float a00 = (1.f - fu) * (1.f - fv), a01 = fu * (1.f - fv), a10 = (1.f - fu) * fv, a11 = fu * fv;
                        for (int p = 0; p < P; ++p) {
                            psf[p] = a00 * k00[p] + a01 * k01[p] + a10 * k10[p] + a11 * k11[p];
                        }
                        // padded pixel (i, j) lands on image pixel (i - R, j - R); kernel pixel
                        // (kx, ky) is offset (kx - R, ky - R)
                        for (int ky = 0; ky < S; ++ky) {
                            const int oy = j - R + ky - R;
                            if ( (oy < oy0) || (oy >= oy1) ) {
                                continue;
                            }
                            for (int kx = 0; kx < S; ++kx) {
                                const int ox = i - R + kx - R;
                                if ( (ox < 0) || (ox >= W) ) {
                                    continue;
                                }
                                const double e = (double)v * psf[(size_t)ky * S + kx];
                                const size_t o = (size_t)oy * W + ox;
                                acc[0][o] += plan.spectral.response[0][l] * e;
                                acc[1][o] += plan.spectral.response[1][l] * e;
                                acc[2][o] += plan.spectral.response[2][l] * e;
                            }
                        }
                    }
                }
            });
        }
        for (std::thread& t : th) {
            t.join();
        }
    }
    for (int c = 0; c < 3; ++c) {
        for (size_t p = 0; p < acc[c].size(); ++p) {
            out.c[c][p] = (float)acc[c][p];
        }
    }
}

} // namespace ca
