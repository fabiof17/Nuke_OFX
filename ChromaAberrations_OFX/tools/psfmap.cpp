// ChromaAberrations_OFX - psfmap: computes the PSF map of a parametric lens and writes it as EXR
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
//   psfmap [parameters.txt] [key=value ...]
//
// The parameter file holds key = value lines (# starts a comment); key=value arguments on the
// command line override it. Run "psfmap --help" for the list of keys.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "exrwriter.h"
#include "psfbank.h"
#include "params.h"

using namespace ca;

static const char* kHelpHead =
    "psfmap - PSF map of a parametric lens (ChromaAberrations_OFX)\n"
    "\n"
    "usage: psfmap [parameters.txt] [key=value ...]\n"
    "\n";
static const char* kHelpTail =
    "Image:\n"
    "  width=2048 height=1152\n"
    "Computation:\n"
    "  kernel=65 (PSF size in pixels)  grid_x=7 grid_y=5  wavelengths=8\n"
    "Output:\n"
    "  output=psfmap.exr  report=psfmap.csv (empty = none)\n"
    "  normalize=cell|global  gain=1  chromatic_shift=1 (show lateral colour)\n"
    "  spectral_channels=0 (1 = also write one channel per wavelength)\n";

using namespace catool;

int main(int argc, char** argv)
{
    Params p;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ( (a == "-h") || (a == "--help") ) {
            std::printf("%s%s%s", kHelpHead, kLensHelp, kHelpTail);
            return 0;
        }
        if (a.find('=') != std::string::npos) {
            parseLine(a, p);
        } else {
            files.push_back(a);
        }
    }
    for (const std::string& f : files) {
        if ( !readParamFile("psfmap", f, p) ) {
            return 1;
        }
    }
    if ( !checkKeys("psfmap", p, { "width", "height", "kernel", "grid_x", "grid_y", "wavelengths", "output", "report",
                                   "normalize", "gain", "chromatic_shift", "spectral_channels",
                                   // carender keys, so that the same lens files work with both tools
                                   "colorspace", "blur", "distortion", "vignette",
                                   "natural_vignetting", "device", "compare", "exact", "save_input" }) ) {
        return 1;
    }

    WavefrontParams w;
    ApertureParams a;
    ImageGeometry g;
    PsfOptions opt;
    lensFromParams(p, w, a, g, opt);
    g.width = (int)num(p, "width", 2048);
    g.height = (int)num(p, "height", 1152);
    const int S = ( (int)num(p, "kernel", 65) ) | 1;
    const int gridX = (int)num(p, "grid_x", 7), gridY = (int)num(p, "grid_y", 5);
    const int nl = (int)num(p, "wavelengths", 8);

    std::printf("Lens: f/%.2f, %d x %d px, pixel %.2f um, kernel %d px, grid %d x %d, %d wavelengths\n",
                w.fNumber, g.width, g.height, g.pixelPitchUm(), S, gridX, gridY, nl);
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<LensModel> model = lensModelFromParams("psfmap", p, w, a);
    if (!model) {
        return 1;
    }
    const std::vector<WavelengthSample> wls = sampleWavelengths(nl);
    {
        std::vector<double> nms;
        for (const WavelengthSample& ws : wls) nms.push_back(ws.nm);
        std::string perr;
        if ( !model->prepare(g, nms, std::function<void(int, int)>(), &perr) ) {
            std::fprintf(stderr, "psfmap: %s\n", perr.c_str());
            return 1;
        }
    }
    PsfBank bank = computePsfBank(*model, g, S, gridX, gridY, wls, opt, [](int d, int t) {
        if ( (d == t) || (d % 8 == 0) ) {
            std::printf("\r  PSF %d / %d", d, t);
            std::fflush(stdout);
        }
    });
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n  computed in %.2f s\n", secs);

    // ---- composite map: one cell per grid point, spectral PSFs turned into colour
    const int border = 4;
    const int C = S + border;
    const int W = gridX * C + border, H = gridY * C + border;
    std::vector<float> R( (size_t)W * H, 0.05f ), G( (size_t)W * H, 0.05f ), B( (size_t)W * H, 0.05f );
    const XYZWeights xw = whiteBalancedXYZWeights(bank.wavelengths);
    const bool shift = num(p, "chromatic_shift", 1.) != 0.;
    const bool global = str(p, "normalize", "cell") == "global";
    const double gain = num(p, "gain", 1.);
    std::vector<std::vector<double> > cellX, cellY, cellZ;
    double globalMax = 0.;
    for (int gy = 0; gy < gridY; ++gy) {
        for (int gx = 0; gx < gridX; ++gx) {
            const int cell = gy * gridX + gx;
            std::vector<double> X( (size_t)S * S, 0. ), Y( (size_t)S * S, 0. ), Z( (size_t)S * S, 0. );
            double refDx = 0., refDy = 0.;
            imageDisplacementUm(w, bank.fieldX[cell], bank.fieldY[cell], 550., refDx, refDy);
            refDx /= g.pixelPitchUm();
            refDy /= g.pixelPitchUm();
            for (int l = 0; l < nl; ++l) {
                const PsfResult& r = bank.at(gx, gy, l);
                const double sx = shift ? r.dispXPx - refDx : 0., sy = shift ? r.dispYPx - refDy : 0.;
                for (int y = 0; y < S; ++y) {
                    for (int x = 0; x < S; ++x) {
                        // bilinear sample of the PSF moved by (sx, sy)
                        const double fx = x - sx, fy = y - sy;
                        const int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
                        const double tx = fx - ix, ty = fy - iy;
                        double v = 0.;
                        for (int dy = 0; dy < 2; ++dy) {
                            for (int dx = 0; dx < 2; ++dx) {
                                const int xx = ix + dx, yy = iy + dy;
                                if ( (xx < 0) || (yy < 0) || (xx >= S) || (yy >= S) ) {
                                    continue;
                                }
                                v += r.kernel[(size_t)yy * S + xx] * (dx ? tx : 1. - tx) * (dy ? ty : 1. - ty);
                            }
                        }
                        X[(size_t)y * S + x] += v * xw.x[l];
                        Y[(size_t)y * S + x] += v * xw.y[l];
                        Z[(size_t)y * S + x] += v * xw.z[l];
                    }
                }
            }
            globalMax = std::max( globalMax, *std::max_element( Y.begin(), Y.end() ) );
            cellX.push_back(X);
            cellY.push_back(Y);
            cellZ.push_back(Z);
        }
    }
    for (int gy = 0; gy < gridY; ++gy) {
        for (int gx = 0; gx < gridX; ++gx) {
            const int cell = gy * gridX + gx;
            const double mx = global ? globalMax : *std::max_element( cellY[cell].begin(), cellY[cell].end() );
            const double k = mx > 0. ? gain / mx : 0.;
            for (int y = 0; y < S; ++y) {
                for (int x = 0; x < S; ++x) {
                    double r, gg, bb;
                    const size_t i = (size_t)y * S + x;
                    xyzToLinearSRGB(cellX[cell][i] * k, cellY[cell][i] * k, cellZ[cell][i] * k, r, gg, bb);
                    // image rows go top-down in the EXR, the grid and the kernels are y-up
                    const int ox = border + gx * C + x;
                    const int oy = H - 1 - (border + gy * C + y);
                    const size_t o = (size_t)oy * W + ox;
                    R[o] = (float)r;
                    G[o] = (float)gg;
                    B[o] = (float)bb;
                }
            }
        }
    }
    std::vector<ExrChannel> channels = { { "R", R.data() }, { "G", G.data() }, { "B", B.data() } };
    std::vector<std::vector<float> > spectral;
    std::vector<std::string> names;
    if (num(p, "spectral_channels", 0.) != 0.) {
        spectral.resize(nl, std::vector<float>( (size_t)W * H, 0.f ));
        for (int l = 0; l < nl; ++l) {
            char name[64];
            std::snprintf(name, sizeof(name), "lambda.%03dnm.Y", (int)std::lround(bank.wavelengths[l].nm));
            names.push_back(name);
            for (int gy = 0; gy < gridY; ++gy) {
                for (int gx = 0; gx < gridX; ++gx) {
                    const PsfResult& r = bank.at(gx, gy, l);
                    for (int y = 0; y < S; ++y) {
                        for (int x = 0; x < S; ++x) {
                            const int ox = border + gx * C + x, oy = H - 1 - (border + gy * C + y);
                            spectral[l][(size_t)oy * W + ox] = r.kernel[(size_t)y * S + x];
                        }
                    }
                }
            }
        }
        for (int l = 0; l < nl; ++l) {
            channels.push_back( { names[l], spectral[l].data() } );
        }
    }
    const std::string out = str(p, "output", "psfmap.exr");
    std::string err;
    if ( !writeExr(out, W, H, channels, &err) ) {
        std::fprintf(stderr, "psfmap: %s\n", err.c_str());
        return 1;
    }
    std::printf("Wrote %s (%d x %d, %d channels)\n", out.c_str(), W, H, (int)channels.size());

    // ---- report
    const std::string rep = str(p, "report", "psfmap.csv");
    if ( !rep.empty() ) {
        FILE* f = std::fopen(rep.c_str(), "w");
        if (f) {
            std::fprintf(f, "grid_x,grid_y,pixel_x,pixel_y,field_x,field_y,lambda_nm,method,pupil_samples,fft_size,"
                            "energy_in_kernel,ee80_px,centroid_x_px,centroid_y_px,displacement_x_px,displacement_y_px\n");
            int nWave = 0, nHybrid = 0;
            double minEnergy = 1.;
            for (int gy = 0; gy < gridY; ++gy) {
                for (int gx = 0; gx < gridX; ++gx) {
                    const int cell = gy * gridX + gx;
                    for (int l = 0; l < nl; ++l) {
                        const PsfResult& r = bank.at(gx, gy, l);
                        (r.method == "wave" ? nWave : nHybrid)++;
                        minEnergy = std::min(minEnergy, r.energyInKernel);
                        std::fprintf(f, "%d,%d,%.1f,%.1f,%.4f,%.4f,%.1f,%s,%d,%d,%.6f,%.3f,%.4f,%.4f,%.4f,%.4f\n",
                                     gx, gy, bank.pixelX[cell], bank.pixelY[cell], bank.fieldX[cell], bank.fieldY[cell],
                                     bank.wavelengths[l].nm, r.method.c_str(), r.pupilSamples, r.fftSize, r.energyInKernel,
                                     r.ee80Px, r.centroidXPx, r.centroidYPx, r.dispXPx, r.dispYPx);
                    }
                }
            }
            std::fclose(f);
            std::printf("Wrote %s (%d wave, %d hybrid PSFs; lowest energy inside the kernel %.1f %%)\n",
                        rep.c_str(), nWave, nHybrid, 100. * minEnergy);
            if (minEnergy < 0.9) {
                std::printf("  note: some PSFs are larger than the kernel, increase kernel= to keep their light\n");
            }
        }
    }
    return 0;
}
