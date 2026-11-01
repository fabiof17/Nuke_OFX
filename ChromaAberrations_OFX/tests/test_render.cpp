// ChromaAberrations_OFX - validation tests of the image renderer (step 2)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "render.h"
#include "render_gpu.h"

using namespace ca;

static int gFail = 0, gChecks = 0;
#define CHECK(cond, ...) do { ++gChecks; bool ok_ = (cond); std::printf("  [%s] ", ok_ ? " ok " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); if (!ok_) ++gFail; } while (0)

static double psnr(const ImageRGB& a, const ImageRGB& ref, double* maxDiff = 0)
{
    double mse = 0., peak = 0., md = 0.;
    size_t n = 0;
    for (int c = 0; c < 3; ++c) {
        for (size_t i = 0; i < ref.c[c].size(); ++i) {
            const double d = a.c[c][i] - ref.c[c][i];
            mse += d * d;
            md = std::max(md, std::fabs(d));
            peak = std::max(peak, (double)std::fabs(ref.c[c][i]));
            ++n;
        }
    }
    if (maxDiff) *maxDiff = md;
    mse /= std::max<size_t>(1, n);
    return mse > 0. ? 10. * std::log10(peak * peak / mse) : 999.;
}

// a test image with edges, gradients, HDR points, saturated and negative values
static ImageRGB testImage(int w, int h)
{
    ImageRGB im;
    im.resize(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t o = (size_t)y * w + x;
            const bool check = ( (x / 12 + y / 12) % 2 ) != 0;
            im.c[0][o] = check ? 0.8f : 0.05f;
            im.c[1][o] = 0.3f + 0.25f * std::sin(x * 0.07f) * std::cos(y * 0.05f);
            im.c[2][o] = (x > w / 2) ? 0.6f : 0.1f;
            if ( (x % 37 == 18) && (y % 29 == 14) ) {
                im.c[0][o] += 40.f; im.c[1][o] += 40.f; im.c[2][o] += 40.f;
            }
        }
    }
    im.c[1][(size_t)(h / 3) * w + w / 5] = -0.2f;   // out-of-gamut colours exist in ACEScg plates
    return im;
}

static void vintageLens(WavefrontParams& w, ApertureParams& a)
{
    w.fNumber = 2.;
    w.W040 = 4.; w.W131 = 10.; w.W222 = 6.; w.W220 = 2.; w.W311 = -2.;
    w.axialCA = 2.; w.lateralCA = 3.;
    a.blades = 6; a.roundness = 0.4; a.vignetting = 0.5;
}

static ImageGeometry smallGeom(int w, int h)
{
    ImageGeometry g;
    g.width = w;
    g.height = h;
    g.filmbackWidthMm = 24.89 * w / 2048. * 2.;   // 24 um pixels: a few pixels of blur
    return g;
}

// ---------------------------------------------------------------------------
static void testCameras()
{
    std::printf("Camera sensitivities (rawtoaces-data, %d cameras)\n", builtinCameraCount());
    double worstRt = 0., worstFit = 0., sumFit = 0.;
    std::string worstName;
    for (int k = 0; k < builtinCameraCount(); ++k) {
        CameraSensitivity cam;
        builtinCamera(builtinCameraId(k), cam);
        const SpectralBasis b = buildSpectralBasis(16, eColorSpaceACEScg, 400., 700., &cam);
        unsigned seed = 777u + k;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.; };
        for (int i = 0; i < 500; ++i) {
            const double rgb[3] = { rnd() * 3., rnd() * 3., rnd() * 3. };
            double back[3] = { 0., 0., 0. };
            for (int l = 0; l < b.count(); ++l) {
                const double sv = spectrumOf(b, l, rgb[0], rgb[1], rgb[2]);
                for (int c = 0; c < 3; ++c) back[c] += b.response[c][l] * sv;
            }
            for (int c = 0; c < 3; ++c) worstRt = std::max(worstRt, std::fabs(back[c] - rgb[c]) / (rgb[0] + rgb[1] + rgb[2]));
        }
        sumFit += b.cameraFitError;
        if (b.cameraFitError > worstFit) {
            worstFit = b.cameraFitError;
            worstName = cam.name;
        }
    }
    CHECK(worstRt < 1e-6, "round trip stays the identity with every camera: largest relative error %.2e", worstRt);
    CHECK(worstFit < 0.05, "camera matrices fitted on 190 colours: RMS error %.1f %% of white on average, %.1f %% at worst (%s)",
          100. * sumFit / builtinCameraCount(), 100. * worstFit, worstName.c_str());
    // a camera changes the colour of single wavelengths (fringes), not of the plate
    CameraSensitivity cam;
    builtinCamera("Sony_ILCE-7SM3", cam);
    const SpectralBasis cie = buildSpectralBasis(16, eColorSpaceACEScg), sony = buildSpectralBasis(16, eColorSpaceACEScg, 400., 700., &cam);
    double diff = 0.;
    for (int l = 0; l < 16; ++l) {
        double a = 0., c = 0.;
        for (int ch = 0; ch < 3; ++ch) { a += cie.response[ch][l]; c += sony.response[ch][l]; }
        for (int ch = 0; ch < 3; ++ch) diff = std::max(diff, std::fabs(cie.response[ch][l] / a - sony.response[ch][l] / c));
    }
    CHECK(diff > 0.02, "%s: the colour of a single wavelength differs from the CIE observer by up to %.2f (chromaticity)",
          cam.name.c_str(), diff);
    // custom file in both formats
    const std::string txt = "/tmp/ca_cam_test.txt";
    {
        FILE* f = std::fopen(txt.c_str(), "w");
        std::fprintf(f, "# nm r g b\nnm,r,g,b\n");
        for (size_t i = 0; i < cam.nm.size(); ++i) std::fprintf(f, "%g, %g, %g, %g\n", cam.nm[i], cam.s[0][i], cam.s[1][i], cam.s[2][i]);
        std::fclose(f);
    }
    CameraSensitivity back;
    std::string err;
    const bool ok = loadCameraFile(txt, back, &err);
    double fd = 0.;
    for (int ch = 0; ok && ch < 3; ++ch)
        for (double l = 400.; l <= 700.; l += 2.5) fd = std::max(fd, std::fabs(back.at(ch, l) - cam.at(ch, l)));
    CHECK(ok && fd < 1e-5, "camera text file read back (%s), largest difference %.1e", ok ? "ok" : err.c_str(), fd);
}

// ---------------------------------------------------------------------------
static void testSpectralBasis()
{
    std::printf("Spectral basis (RGB -> spectrum -> RGB)\n");
    double worst = 0., worstGrey = 0., minRec709 = 1.;
    const int counts[4] = { 3, 8, 16, 32 };
    for (int cs = 0; cs < 3; ++cs) {
        for (int n : counts) {
            const SpectralBasis b = buildSpectralBasis(n, (ColorSpace)cs);
            for (int c = 0; c < 3; ++c) {
                for (int d = 0; d < 3; ++d) {
                    double s = 0.;
                    for (int i = 0; i < b.count(); ++i) s += b.response[c][i] * b.basis[d][i];
                    worst = std::max(worst, std::fabs(s - (c == d ? 1. : 0.)));
                }
            }
            for (int i = 0; i < b.count(); ++i) {
                worstGrey = std::max(worstGrey, std::fabs(b.basis[0][i] + b.basis[1][i] + b.basis[2][i] - 1.));
                if ( (cs == eColorSpaceRec709) && (n >= 8) ) {
                    for (int d = 0; d < 3; ++d) minRec709 = std::min(minRec709, b.basis[d][i]);
                }
            }
        }
    }
    CHECK(worst < 1e-9, "round trip is the identity: largest error %.2e (ACEScg, Rec.709, Rec.2020; 3 to 32 wavelengths)", worst);
    CHECK(worstGrey < 1e-9, "greys become flat spectra: largest error %.2e", worstGrey);
    CHECK(minRec709 > -0.02, "Rec.709 colours have non-negative spectra: lowest basis value %.4f", minRec709);

    // per-colour spectra (chromaticity table)
    const SpectralBasis b = buildSpectralBasis(8, eColorSpaceACEScg);
    double rt = 0.;
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.; };
    for (int k = 0; k < 20000; ++k) {
        // HDR, saturated, slightly negative, out-of-table and zero-sum colours
        double rgb[3] = { rnd() * 4. - 0.5, rnd() * 4. - 0.5, rnd() * 4. - 0.5 };
        if (k % 5 == 0) { rgb[k % 3] = 0.; }
        if (k % 7 == 0) { rgb[0] *= 100.; rgb[1] *= 100.; rgb[2] *= 100.; }
        double back[3] = { 0., 0., 0. };
        for (int l = 0; l < b.count(); ++l) {
            const double s = spectrumOf(b, l, rgb[0], rgb[1], rgb[2]);
            for (int c = 0; c < 3; ++c) back[c] += b.response[c][l] * s;
        }
        const double mag = std::fabs(rgb[0]) + std::fabs(rgb[1]) + std::fabs(rgb[2]);
        for (int c = 0; c < 3; ++c) rt = std::max(rt, std::fabs(back[c] - rgb[c]) / std::max(mag, 1e-3));
    }
    CHECK(rt < 1e-6, "round trip with the per-colour spectra: largest relative error %.2e (20000 random colours)", rt);
    const double sat[4][3] = { { 1., 0.05, 0.02 }, { 0.02, 0.05, 1. }, { 1., 0.6, 0.02 }, { 0.9, 0.1, 0.8 } };
    const char* names[4] = { "red neon", "blue LED", "sodium orange", "magenta" };
    for (int k = 0; k < 4; ++k) {
        double mn = 1e9, mx = -1e9;
        for (int l = 0; l < b.count(); ++l) {
            const double s = spectrumOf(b, l, sat[k][0], sat[k][1], sat[k][2]);
            mn = std::min(mn, s); mx = std::max(mx, s);
        }
        double lin = 1e9;
        for (int l = 0; l < b.count(); ++l) lin = std::min(lin, b.basis[0][l] * sat[k][0] + b.basis[1][l] * sat[k][1] + b.basis[2][l] * sat[k][2]);
        CHECK(mn > -0.01 * mx, "ACEScg %s (%.2f, %.2f, %.2f): spectrum >= %.3f x its peak (a fixed linear basis would go down to %.3f)",
              names[k], sat[k][0], sat[k][1], sat[k][2], mn / mx, lin / mx);
    }
}

static void testPipelineIdentity()
{
    std::printf("Renderer with every effect off returns the input\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    rs.blur = rs.distortion = rs.vignetting = false;
    const ImageRGB in = testImage(160, 96);
    for (int cs = 0; cs < 3; ++cs) {
        rs.colorSpace = (ColorSpace)cs;
        LensPlan plan = buildLensPlan(w, a, smallGeom(160, 96), rs);
        ImageRGB out;
        renderCpu(plan, in, out);
        double rel = 0.;
        for (int c = 0; c < 3; ++c) {
            for (size_t i = 0; i < in.c[c].size(); ++i) {
                rel = std::max(rel, std::fabs(out.c[c][i] - in.c[c][i]) / std::max(1.0, (double)std::fabs(in.c[c][i])));
            }
        }
        CHECK(rel < 1e-5, "%s: largest relative error %.2e (HDR and negative values included)", colorSpaceName((ColorSpace)cs), rel);
    }
}

static void testExact()
{
    std::printf("Tiled FFT renderer against the exact per-pixel convolution (vintage lens, 192 x 112, kernel 33, 8 wavelengths)\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    rs.kernelSize = 33;
    const ImageGeometry g = smallGeom(192, 112);
    const ImageRGB in = testImage(192, 112);
    const int grids[2][2] = { { 7, 5 }, { 4, 3 } };
    for (const auto& gr : grids) {
        rs.gridX = gr[0];
        rs.gridY = gr[1];
        LensPlan plan = buildLensPlan(w, a, g, rs);
        ImageRGB ex, out;
        renderExactCpu(plan, in, ex);
        renderCpu(plan, in, out);
        double md;
        const double db = psnr(out, ex, &md);
        CHECK(db > 100., "grid %d x %d: PSNR %.1f dB, max difference %.2e (float FFT precision)", gr[0], gr[1], db, md);
    }
}

// The renderer interpolates the PSF between grid points: compare with the PSF computed exactly
// at the position of isolated point sources placed between the grid points.
static void testGridDensity()
{
    std::printf("PSF between grid points against the PSF computed at that point (vintage lens, 384 x 216, kernel 41)\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    a.vignetting = 0.;
    RenderSettings rs;
    rs.kernelSize = 41;
    rs.distortion = false;
    rs.vignetting = false;
    const int W = 384, H = 216;
    const ImageGeometry g = smallGeom(W, H);
    ImageRGB in;
    in.resize(W, H);
    std::vector<std::pair<int, int> > pts;
    for (int y = 30; y < H - 20; y += 47) {
        for (int x = 25; x < W - 20; x += 59) {
            pts.push_back(std::make_pair(x, y));
            for (int c = 0; c < 3; ++c) in.c[c][(size_t)y * W + x] = 1.f;
        }
    }
    const SpectralBasis sb = buildSpectralBasis(rs.wavelengths, rs.colorSpace);
    SensorParams sp;
    sp.pixelPitchUm = g.pixelPitchUm();
    sp.kernelSize = rs.kernelSize;
    // expected response of each point: sum over wavelengths of the exact PSF
    std::vector<std::vector<double> > expect[3];
    for (int c = 0; c < 3; ++c) expect[c].resize(pts.size());
    for (size_t k = 0; k < pts.size(); ++k) {
        double hx, hy;
        pixelToField(g, pts[k].first + 0.5, pts[k].second + 0.5, hx, hy);
        for (int c = 0; c < 3; ++c) expect[c][k].assign( (size_t)sp.kernelSize * sp.kernelSize, 0. );
        for (int l = 0; l < sb.count(); ++l) {
            PsfResult r = computePsf(w, a, sp, hx, hy, sb.wavelengths[l].nm, rs.psf);
            double s = 0.;
            for (float v : r.kernel) s += v;
            for (int c = 0; c < 3; ++c) {
                for (size_t p = 0; p < r.kernel.size(); ++p) {
                    expect[c][k][p] += sb.response[c][l] * r.kernel[p] / s;
                }
            }
        }
    }
    const int grids[3][2] = { { 4, 3 }, { 7, 5 }, { 15, 11 } };
    double errs[3];
    for (int q = 0; q < 3; ++q) {
        rs.gridX = grids[q][0];
        rs.gridY = grids[q][1];
        LensPlan plan = buildLensPlan(w, a, g, rs);
        ImageRGB out;
        renderCpu(plan, in, out);
        double num = 0., den = 0.;
        const int S = rs.kernelSize, R = S / 2;
        for (size_t k = 0; k < pts.size(); ++k) {
            for (int ky = 0; ky < S; ++ky) for (int kx = 0; kx < S; ++kx) {
                const int ox = pts[k].first + kx - R, oy = pts[k].second + ky - R;
                if ( (ox < 0) || (ox >= W) || (oy < 0) || (oy >= H) ) continue;
                // only this point's light: the points are farther apart than the kernel
                const double e = expect[1][k][(size_t)ky * S + kx];
                num += std::fabs(out.c[1][(size_t)oy * W + ox] - e);
                den += std::fabs(e);
            }
        }
        errs[q] = num / den;
        std::printf("         grid %2d x %2d: PSF error %.1f %% (L1, green channel)\n", grids[q][0], grids[q][1], 100. * errs[q]);
    }
    CHECK(errs[2] < errs[1] && errs[1] < errs[0], "a finer grid follows the lens more closely (%.1f %% -> %.1f %% -> %.1f %%)",
          100. * errs[0], 100. * errs[1], 100. * errs[2]);
}

static void testPointAtNode()
{
    std::printf("A point source on a grid point is spread exactly by that grid point's PSFs\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    rs.kernelSize = 33;
    rs.distortion = false;
    rs.vignetting = false;
    const int W = 192, H = 112;
    LensPlan plan = buildLensPlan(w, a, smallGeom(W, H), rs);
    // grid node (gx, gy) = (5, 1) of a 7 x 5 grid sits at pixel position (W * 5 / 6, H * 1 / 4)
    const int gx = 5, gy = 1;
    const int px = W * gx / 6, py = H * gy / 4;      // pixel whose lower-left corner is the node
    ImageRGB in;
    in.resize(W, H);
    for (int c = 0; c < 3; ++c) in.c[c][(size_t)py * W + px] = 1.f;
    ImageRGB out;
    renderCpu(plan, in, out);
    // expected: sum over wavelengths of response * (flat spectrum) * PSF at the pixel centre,
    // i.e. bilinear between the 4 nodes around (px + 0.5, py + 0.5)
    double md = 0.;
    const int S = plan.S, R = plan.R, P = S * S;
    double u, v;
    plan.gridCoords(px + 0.5, py + 0.5, u, v);
    const int i0 = std::min((int)u, plan.gridX - 2), j0 = std::min((int)v, plan.gridY - 2);
    const double fu = u - i0, fv = v - j0;
    for (int c = 0; c < 3; ++c) {
        for (int ky = 0; ky < S; ++ky) {
            for (int kx = 0; kx < S; ++kx) {
                double e = 0.;
                for (int l = 0; l < plan.wavelengthCount(); ++l) {
                    const std::vector<float>& bk = plan.bankKernels[l];
                    const size_t k = (size_t)ky * S + kx;
                    const double psf = (1. - fu) * (1. - fv) * bk[(size_t)(j0 * plan.gridX + i0) * P + k] +
                                       fu * (1. - fv) * bk[(size_t)(j0 * plan.gridX + i0 + 1) * P + k] +
                                       (1. - fu) * fv * bk[(size_t)((j0 + 1) * plan.gridX + i0) * P + k] +
                                       fu * fv * bk[(size_t)((j0 + 1) * plan.gridX + i0 + 1) * P + k];
                    e += plan.spectral.response[c][l] * psf;
                }
                const int ox = px + kx - R, oy = py + ky - R;
                if ( (ox >= 0) && (ox < W) && (oy >= 0) && (oy < H) ) {
                    md = std::max(md, std::fabs(out.c[c][(size_t)oy * W + ox] - e));
                }
            }
        }
    }
    CHECK(md < 1e-5, "largest difference to the grid PSFs: %.2e", md);
}

static void testEnergy()
{
    std::printf("Energy conservation\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    rs.kernelSize = 33;
    rs.distortion = false;
    rs.vignetting = false;
    // a bright patch away from the borders keeps all its light, for every truncation of the basis
    const int W = 160, H = 96;
    ImageRGB in;
    in.resize(W, H);
    for (int y = 30; y < 66; ++y) for (int x = 40; x < 120; ++x) {
        in.c[0][(size_t)y * W + x] = 2.f; in.c[1][(size_t)y * W + x] = 1.f; in.c[2][(size_t)y * W + x] = 0.25f;
    }
    double worst = 0.;
    {
        LensPlan plan = buildLensPlan(w, a, smallGeom(W, H), rs);
        ImageRGB out;
        renderCpu(plan, in, out);
        for (int c = 0; c < 3; ++c) {
            double si = 0., so = 0.;
            for (size_t i = 0; i < in.c[c].size(); ++i) { si += in.c[c][i]; so += out.c[c][i]; }
            worst = std::max(worst, std::fabs(so / si - 1.));
        }
    }
    CHECK(worst < 1e-5, "total light of each channel kept within %.2e", worst);
    // a lens whose PSF does not vary over the field keeps a uniform image uniform
    WavefrontParams wu;
    wu.fNumber = 2.;
    wu.W040 = 4.;
    wu.W020 = -3.;
    wu.axialCA = 2.;
    ApertureParams au;
    au.blades = 7;
    for (int c = 0; c < 3; ++c) std::fill(in.c[c].begin(), in.c[c].end(), 0.5f);
    LensPlan plan = buildLensPlan(wu, au, smallGeom(W, H), rs);
    ImageRGB out;
    renderCpu(plan, in, out);
    double md = 0.;
    for (int c = 0; c < 3; ++c) for (float v : out.c[c]) md = std::max(md, std::fabs(v - 0.5));
    CHECK(md < 1e-4, "field-independent lens on a uniform image: largest deviation from 0.5 is %.2e", md);
    // (with coma or astigmatism the PSF centroid moves across the field, which acts like a slight
    // distortion: a uniform image then gets small real brightness variations)
}

static void testLateralColour()
{
    std::printf("Distortion and lateral colour move each wavelength where the lens model says\n");
    WavefrontParams w;
    w.fNumber = 4.;
    w.W311 = -1.5;
    w.lateralCA = 4.;
    ApertureParams a;
    RenderSettings rs;
    rs.blur = false;
    rs.vignetting = false;
    const int W = 320, H = 180;
    const ImageGeometry g = smallGeom(W, H);
    LensPlan plan = buildLensPlan(w, a, g, rs);
    ImageRGB in;
    in.resize(W, H);
    const int px = 280, py = 150;
    for (int c = 0; c < 3; ++c) in.c[c][(size_t)py * W + px] = 1.f;
    ImageRGB out;
    renderCpu(plan, in, out);
    double worst = 0.;
    double hx, hy;
    pixelToField(g, px + 0.5, py + 0.5, hx, hy);
    for (int c = 0; c < 3; ++c) {
        double s = 0., sx = 0., sy = 0.;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const double v = out.c[c][(size_t)y * W + x];
            s += v; sx += v * (x + 0.5); sy += v * (y + 0.5);
        }
        // expected: response-weighted chief ray positions
        double ex = 0., ey = 0., ws = 0.;
        for (int l = 0; l < plan.wavelengthCount(); ++l) {
            double dx, dy;
            imageDisplacementUm(w, hx, hy, plan.spectral.wavelengths[l].nm, dx, dy);
            const double wl = plan.spectral.response[c][l];
            ex += wl * (px + 0.5 + dx / g.pixelPitchUm());
            ey += wl * (py + 0.5 + dy / g.pixelPitchUm());
            ws += wl;
        }
        worst = std::max(worst, std::hypot(sx / s - ex / ws, sy / s - ey / ws));
    }
    CHECK(worst < 0.05, "channel centroids within %.3f px of the model", worst);
}

static void testAnamorphic()
{
    std::printf("Anamorphic plate (pixel aspect 2): lens modelled in the desqueezed space\n");
    WavefrontParams w;
    w.fNumber = 2.8;
    w.W020 = 40.;          // defocus: the PSF is the (round) pupil, ~10 px radius
    w.W311 = -2.;
    w.lateralCA = 3.;
    ApertureParams a;
    RenderSettings rs;
    rs.kernelSize = 41;
    rs.vignetting = false;
    rs.wavelengths = 6;
    const int W = 240, H = 200;
    ImageGeometry g = smallGeom(W, H);
    g.pixelAspect = 2.;
    LensPlan plan = buildLensPlan(w, a, g, rs);
    // centre PSF: a disc in the desqueezed space, half as wide on the plate
    const int S = plan.S, P = S * S, cell = (plan.gridY / 2) * plan.gridX + plan.gridX / 2;
    const std::vector<float>& k = plan.bankKernels[plan.wavelengthCount() / 2];
    double s = 0., mx = 0., my = 0., vx = 0., vy = 0.;
    for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
        const double v = k[(size_t)cell * P + (size_t)y * S + x];
        s += v; mx += v * x; my += v * y;
    }
    mx /= s; my /= s;
    for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
        const double v = k[(size_t)cell * P + (size_t)y * S + x];
        vx += v * (x - mx) * (x - mx); vy += v * (y - my) * (y - my);
    }
    // a uniform disc of radius r on pixels a x 1: variance (r/a)^2/4 + 1/12 in x, r^2/4 + 1/12 in y
    const double r2 = 4. * (vy / s - 1. / 12.);
    const double expectVx = r2 / 4. / 4. + 1. / 12.;
    CHECK(std::fabs(vx / s / expectVx - 1.) < 0.05, "round defocus disc squeezed 2x on the plate: x variance %.2f px^2 (expected %.2f)", vx / s, expectVx);
    // chief ray displacement, in plate pixels
    rs.blur = false;
    plan = buildLensPlan(w, a, g, rs);
    ImageRGB in;
    in.resize(W, H);
    const int px = 30, py = 170;
    for (int c = 0; c < 3; ++c) in.c[c][(size_t)py * W + px] = 1.f;
    ImageRGB out;
    renderCpu(plan, in, out);
    double hx, hy;
    pixelToField(g, px + 0.5, py + 0.5, hx, hy);
    double worst = 0.;
    for (int c = 0; c < 3; ++c) {
        double ss = 0., sx = 0., sy = 0.;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const double v = out.c[c][(size_t)y * W + x];
            ss += v; sx += v * (x + 0.5); sy += v * (y + 0.5);
        }
        double ex = 0., ey = 0., ws = 0.;
        for (int l = 0; l < plan.wavelengthCount(); ++l) {
            double dx, dy;
            imageDisplacementUm(w, hx, hy, plan.spectral.wavelengths[l].nm, dx, dy);
            const double wl = plan.spectral.response[c][l];
            ex += wl * (px + 0.5 + dx / (g.pixelPitchUm() * 2.));
            ey += wl * (py + 0.5 + dy / g.pixelPitchUm());
            ws += wl;
        }
        worst = std::max(worst, std::hypot(sx / ss - ex / ws, sy / ss - ey / ws));
    }
    CHECK(worst < 0.05, "distortion and lateral colour on the squeezed plate: within %.3f px of the model", worst);
}

static void testVignetting()
{
    std::printf("Relative illumination\n");
    WavefrontParams w;
    ApertureParams a;
    RenderSettings rs;
    rs.blur = false;
    rs.distortion = false;
    rs.focalLengthMm = 24.;
    const int W = 256, H = 144;
    ImageGeometry g;
    g.width = W;
    g.height = H;     // Super 35 width
    ImageRGB in;
    in.resize(W, H);
    for (int c = 0; c < 3; ++c) std::fill(in.c[c].begin(), in.c[c].end(), 1.f);
    LensPlan plan = buildLensPlan(w, a, g, rs);
    ImageRGB out;
    renderCpu(plan, in, out);
    const double rmm = std::hypot(W / 2. - 0.5, H / 2. - 0.5) * g.pixelPitchUm() * 1e-3;
    const double c4 = std::pow(std::cos(std::atan(rmm / 24.)), 4.);
    CHECK(std::fabs(out.c[1][0] - c4) < 2e-3, "natural falloff in the corner: %.4f (cos^4 law: %.4f)", out.c[1][0], c4);
    a.vignetting = 0.5;
    rs.naturalVignetting = 0.;
    plan = buildLensPlan(w, a, g, rs);
    renderCpu(plan, in, out);
    // overlap of the pupil (radius 1) and the clipping circle (radius rv, k |H| away)
    double hx, hy, rv, k;
    pixelToField(g, 0.5, 0.5, hx, hy);
    vignettingCircle(a, rv, k);
    const double d = k * std::hypot(hx, hy);
    const double lens = std::acos( (d * d + 1. - rv * rv) / (2. * d) ) + rv * rv * std::acos( (d * d + rv * rv - 1.) / (2. * d * rv) ) -
                        0.5 * std::sqrt( (-d + 1. + rv) * (d + 1. - rv) * (d - 1. + rv) * (d + 1. + rv) );
    const double expect = lens / 3.14159265358979;
    const double centre = out.c[1][(size_t)(H / 2) * W + W / 2];
    CHECK(std::fabs(out.c[1][0] - expect) < 0.01 && std::fabs(centre - 1.) < 0.01,
          "mechanical vignetting: corner %.4f (overlap of pupil and barrel: %.4f), centre %.4f", out.c[1][0], expect, centre);
}

static void testGpu()
{
    std::printf("GPU renderer against the CPU reference\n");
    std::string name;
    if ( !GpuRenderer::available(&name) ) {
        std::printf("  [skip] %s\n", name.c_str());
        return;
    }
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    rs.kernelSize = 33;
    const ImageGeometry g = smallGeom(320, 180);
    const ImageRGB in = testImage(320, 180);
    LensPlan plan = buildLensPlan(w, a, g, rs);
    ImageRGB cpu, gpu1, gpu2;
    renderCpu(plan, in, cpu);
    GpuRenderer r;
    std::string err;
    bool ok = r.setPlan(plan, &err) && r.render(in, gpu1, 0, &err);
    CHECK(ok, "%s: render %s", name.c_str(), ok ? "done" : err.c_str());
    if (!ok) return;
    double md;
    double db = psnr(gpu1, cpu, &md);
    CHECK(db > 90., "same image: PSNR %.1f dB, max difference %.2e", db, md);
    RenderStats st;
    ok = r.render(in, gpu2, &st, &err);
    db = psnr(gpu2, cpu, &md);
    CHECK(ok && db > 90., "second frame (PSF spectra %s): PSNR %.1f dB", st.kernelCache ? "kept on the GPU" : "recomputed", db);
    // page-locked buffers (the path used by the plugin)
    {
        float* hin = r.inputBuffer(in.width, in.height);
        const size_t n = (size_t)in.width * in.height;
        ok = hin != 0;
        if (ok) {
            for (int c = 0; c < 3; ++c) std::copy(in.c[c].begin(), in.c[c].end(), hin + c * n);
            ok = r.renderBuffers(in.width, in.height, &st, &err);
        }
        ImageRGB b;
        b.resize(in.width, in.height);
        if (ok) for (int c = 0; c < 3; ++c) std::copy(r.outputBuffer() + c * n, r.outputBuffer() + (c + 1) * n, b.c[c].begin());
        db = ok ? psnr(b, cpu, &md) : 0.;
        CHECK(ok && db > 90., "page-locked transfer buffers: PSNR %.1f dB", db);
    }
    r.setKernelCacheBudget(0);
    ok = r.render(in, gpu2, &st, &err);
    db = psnr(gpu2, cpu, &md);
    CHECK(ok && !st.kernelCache && db > 90., "without the PSF spectra cache: PSNR %.1f dB", db);
    r.setWavelengthBatch(3);   // as when the GPU memory is short (Final in 4K): 8 wavelengths in 3 batches
    ok = r.render(in, gpu2, &st, &err);
    db = psnr(gpu2, cpu, &md);
    CHECK(ok && st.batches == 3 && db > 90., "wavelengths in %d batches: PSNR %.1f dB", st.batches, db);
    r.setKernelCacheBudget((size_t)-1);
    ok = r.render(in, gpu2, &st, &err) && r.render(in, gpu2, &st, &err);
    db = psnr(gpu2, cpu, &md);
    CHECK(ok && st.batches == 3 && db > 90., "batches with the PSF spectra cache (%s): PSNR %.1f dB", st.kernelCache ? "kept" : "not kept", db);
    r.setWavelengthBatch(0);
    rs.blur = false;
    plan = buildLensPlan(w, a, g, rs);
    renderCpu(plan, in, cpu);
    ok = r.setPlan(plan, &err) && r.render(in, gpu1, 0, &err);
    db = psnr(gpu1, cpu, &md);
    CHECK(ok && db > 100., "distortion, colour and vignetting only (no blur): PSNR %.1f dB", db);
}

static void testTiming()
{
    std::printf("Timing at 2K (2048 x 1152, kernel 65, grid 7 x 5, 8 wavelengths)\n");
    WavefrontParams w;
    ApertureParams a;
    vintageLens(w, a);
    RenderSettings rs;
    ImageGeometry g;
    LensPlan plan = buildLensPlan(w, a, g, rs);
    std::printf("         lens plan: %.2f s\n", plan.buildSeconds);
    const ImageRGB in = testImage(g.width, g.height);
    ImageRGB out;
    RenderStats st;
    renderCpu(plan, in, out, &st);
    std::printf("         CPU: %.2f s per frame (tiles %d x %d, %d tile convolutions)\n", st.seconds, st.fftWidth, st.fftHeight, st.convolutions);
    std::string name;
    if (GpuRenderer::available(&name)) {
        GpuRenderer r;
        std::string err;
        if (r.setPlan(plan, &err) && r.render(in, out, &st, &err)) {
            std::printf("         GPU %s, first frame: %.1f ms (GPU %.1f ms, transfers %.1f ms)\n", name.c_str(), st.seconds * 1e3, st.fftSeconds * 1e3, st.transferSeconds * 1e3);
            for (int i = 0; i < 3; ++i) r.render(in, out, &st, &err);
            std::printf("         GPU, next frames: %.1f ms (GPU %.1f ms, transfers %.1f ms, PSF spectra %s)\n", st.seconds * 1e3, st.fftSeconds * 1e3,
                        st.transferSeconds * 1e3, st.kernelCache ? "kept on the GPU" : "recomputed");
        }
    }
}

int main(int argc, char** argv)
{
    const bool timing = !(argc > 1 && std::string(argv[1]) == "--no-timing");
    testSpectralBasis();
    testCameras();
    testPipelineIdentity();
    testExact();
    testGridDensity();
    testPointAtNode();
    testEnergy();
    testLateralColour();
    testVignetting();
    testAnamorphic();
    testGpu();
    if (timing) testTiming();
    std::printf("\n%d checks, %d failures\n", gChecks, gFail);
    return gFail ? 1 : 0;
}
