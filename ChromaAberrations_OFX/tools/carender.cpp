// ChromaAberrations_OFX - carender: applies a parametric lens to an EXR image (step 2 renderer)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
//   carender input.exr output.exr [parameters.txt] [key=value ...]
//   carender pattern:chart output.exr [parameters.txt] [key=value ...]

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "params.h"
#include "render.h"
#include "render_gpu.h"
#include "views.h"

using namespace ca;
using namespace catool;

static const char* kHelpHead =
    "carender - applies a parametric lens to an image (ChromaAberrations_OFX)\n"
    "\n"
    "usage: carender input.exr output.exr [parameters.txt] [key=value ...]\n"
    "       input can also be pattern:chart, pattern:points or pattern:siemens (test images)\n"
    "\n";
static const char* kHelpTail =
    "Rendering:\n"
    "  wavelengths=8  grid_x=7 grid_y=5 (PSFs computed on this grid, interpolated between)  kernel=65\n"
    "  auto_kernel=0 (1: PSF size from the lens, up to kernel)  auto_grid=0 (1: refine the grid from grid_x x grid_y\n"
    "  until the interpolation error is below grid_tolerance=0.02)\n"
    "  colorspace=acescg|rec709|rec2020 (linear working space of the image)\n"
    "  camera=<id>|<file> (camera sensitivities, e.g. Sony_ILCE-7SM3; default: CIE 1931 observer;\n"
    "         camera=list prints the built-in cameras)\n"
    "  blur=1  distortion=1  vignette=1 (each effect can be switched off)\n"
    "  focal=35 (mm, for the natural cos^4 falloff)  natural_vignetting=1 (0..1)\n"
    "  device=auto|cpu|gpu\n"
    "Checks:\n"
    "  compare=1  render on the CPU and the GPU and print the difference\n"
    "  exact=1    also render with the exact per-pixel convolution and print the PSNR (slow)\n"
    "Test images (input pattern:...):\n"
    "  width=2048 height=1152  save_input=pattern.exr\n";

static bool loadExr(const std::string& path, ImageRGB& img, std::vector<float>& alpha)
{
    float* rgba = 0;
    int w = 0, h = 0;
    const char* err = 0;
    if (LoadEXR(&rgba, &w, &h, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::fprintf(stderr, "carender: cannot read %s: %s\n", path.c_str(), err ? err : "unknown error");
        if (err) {
            FreeEXRErrorMessage(err);
        }
        return false;
    }
    img.resize(w, h);
    alpha.assign( (size_t)w * h, 1.f );
    for (int y = 0; y < h; ++y) {
        const float* row = rgba + (size_t)(h - 1 - y) * w * 4;   // EXR is top-down, the renderer bottom-up
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                img.c[c][(size_t)y * w + x] = row[x * 4 + c];
            }
            alpha[(size_t)y * w + x] = row[x * 4 + 3];
        }
    }
    free(rgba);
    return true;
}

static bool saveExr(const std::string& path, const ImageRGB& img, const std::vector<float>& alpha)
{
    const int w = img.width, h = img.height;
    std::vector<float> rgba( (size_t)w * h * 4 );
    for (int y = 0; y < h; ++y) {
        float* row = &rgba[(size_t)(h - 1 - y) * w * 4];
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                row[x * 4 + c] = img.c[c][(size_t)y * w + x];
            }
            row[x * 4 + 3] = alpha.empty() ? 1.f : alpha[(size_t)y * w + x];
        }
    }
    const char* err = 0;
    if (SaveEXR(rgba.data(), w, h, 4, 0, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::fprintf(stderr, "carender: cannot write %s: %s\n", path.c_str(), err ? err : "unknown error");
        if (err) {
            FreeEXRErrorMessage(err);
        }
        return false;
    }
    return true;
}

static double psnr(const ImageRGB& a, const ImageRGB& ref, double* maxDiff)
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
    if (maxDiff) {
        *maxDiff = md;
    }
    mse /= std::max<size_t>(1, n);
    return mse > 0. ? 10. * std::log10(peak * peak / mse) : 999.;
}

int main(int argc, char** argv)
{
    Params p;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ( (a == "-h") || (a == "--help") ) {
            std::printf("%s%s%s", kHelpHead, kLensHelp, kHelpTail);
            return 0;
        }
        if ( (a.find('=') != std::string::npos) && (a.compare(0, 8, "pattern:") != 0) ) {
            parseLine(a, p);
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() < 2) {
        std::printf("%s%s%s", kHelpHead, kLensHelp, kHelpTail);
        return 1;
    }
    for (size_t i = 2; i < pos.size(); ++i) {
        if ( !readParamFile("carender", pos[i], p) ) {
            return 1;
        }
    }
    if ( !checkKeys("carender", p, { "wavelengths", "grid_x", "grid_y", "kernel", "auto_kernel", "auto_grid", "grid_tolerance", "colorspace", "camera", "blur", "distortion",
                                     "vignette", "focal", "natural_vignetting", "device", "compare", "exact", "width", "height",
                                     "save_input",
                                     // psfmap keys, so that the same lens files work with both tools
                                     "output", "report", "normalize", "gain", "chromatic_shift", "spectral_channels" }) ) {
        return 1;
    }

    // ---- input
    ImageRGB in;
    std::vector<float> alpha;
    if (pos[0].compare(0, 8, "pattern:") == 0) {
        makeTestChart(pos[0].substr(8), (int)num(p, "width", 2048), (int)num(p, "height", 1152), in);
        const std::string si = str(p, "save_input", "");
        if ( !si.empty() && !saveExr(si, in, alpha) ) {
            return 1;
        }
    } else if ( !loadExr(pos[0], in, alpha) ) {
        return 1;
    }

    // ---- lens plan
    WavefrontParams w;
    ApertureParams a;
    ImageGeometry g;
    RenderSettings rs;
    lensFromParams(p, w, a, g, rs.psf);
    g.width = in.width;
    g.height = in.height;
    rs.wavelengths = (int)num(p, "wavelengths", 8);
    rs.gridX = (int)num(p, "grid_x", 7);
    rs.gridY = (int)num(p, "grid_y", 5);
    rs.kernelSize = (int)num(p, "kernel", 65) | 1;
    rs.autoKernel = num(p, "auto_kernel", 0.) != 0.;
    rs.autoGrid = num(p, "auto_grid", 0.) != 0.;
    rs.gridTolerance = num(p, "grid_tolerance", 0.02);
    rs.blur = num(p, "blur", 1.) != 0.;
    rs.distortion = num(p, "distortion", 1.) != 0.;
    rs.vignetting = num(p, "vignette", 1.) != 0.;
    rs.focalLengthMm = num(p, "focal", p.count("lens") ? 50. : 35.);
    rs.naturalVignetting = num(p, "natural_vignetting", 1.);
    if ( !colorSpaceFromName(lower(str(p, "colorspace", "acescg")), rs.colorSpace) ) {
        std::fprintf(stderr, "carender: unknown colorspace (acescg, rec709 or rec2020)\n");
        return 1;
    }
    if ( p.count("camera") ) {
        const std::string c = str(p, "camera", "");
        if (c == "list") {
            for (int i = 0; i < builtinCameraCount(); ++i) std::printf("%-28s %s\n", builtinCameraId(i), builtinCameraName(i));
            return 0;
        }
        std::shared_ptr<CameraSensitivity> cam = std::make_shared<CameraSensitivity>();
        std::string err;
        if ( !builtinCamera(c, *cam) && !loadCameraFile(c, *cam, &err) ) {
            std::fprintf(stderr, "carender: camera %s: %s\n", c.c_str(), err.c_str());
            return 1;
        }
        rs.camera = cam;
        std::printf("Camera: %s\n", cam->name.c_str());
    }
    std::printf("Image %d x %d, pixel %.2f um, f/%.2f, %d wavelengths, grid %d x %d, kernel %d, %s\n",
                in.width, in.height, g.pixelPitchUm(), w.fNumber, rs.wavelengths, rs.gridX, rs.gridY, rs.kernelSize,
                colorSpaceName(rs.colorSpace));
    std::unique_ptr<LensModel> model = lensModelFromParams("carender", p, w, a);
    if (!model) {
        return 1;
    }
    LensPlan plan;
    std::string perr;
    if ( !buildLensPlan(*model, g, rs, plan, [](int d, int t) {
        if ( (d == t) || (d % 16 == 0) ) {
            std::printf("\r  %d / %d", d, t);
            std::fflush(stdout);
        }
    }, &perr) ) {
        std::fprintf(stderr, "carender: %s\n", perr.c_str());
        return 1;
    }
    std::printf("\n  lens plan: %.2f s (lens %.2f s, PSFs %.2f s, maps %.2f s), grid %d x %d%s, kernel %d, %d PSFs, at least %.1f %% "
                "of each PSF inside the kernel, max displacement %.2f px\n",
                plan.buildSeconds, plan.prepareSeconds, plan.psfSeconds, plan.mapsSeconds, plan.gridX, plan.gridY,
                plan.gridError >= 0. ? (" (interpolation error " + std::to_string((int)std::lround(plan.gridError * 1000.) / 10.).substr(0, 4) + " %)").c_str() : "",
                plan.S, plan.nodeCount() * plan.wavelengthCount(), 100. * plan.minEnergyInKernel, plan.maxDisplacementPx);
    if (plan.minEnergyInKernel < 0.97) {
        std::printf("  warning: some PSFs are larger than the kernel, increase kernel= for a faithful blur\n");
    }

    // ---- render
    const std::string device = lower(str(p, "device", "auto"));
    const bool compare = num(p, "compare", 0.) != 0.;
    std::string gpuName;
    const bool haveGpu = GpuRenderer::available(&gpuName);
    if ( (device == "gpu") && !haveGpu ) {
        std::fprintf(stderr, "carender: no GPU (%s)\n", gpuName.c_str());
        return 1;
    }
    ImageRGB out, outCpu;
    RenderStats st;
    bool rendered = false;
    if ( haveGpu && (device != "cpu") ) {
        GpuRenderer gpu;
        std::string err;
        if ( gpu.setPlan(plan, &err) && gpu.render(in, out, &st, &err) ) {
            std::printf("GPU %s: %.3f s (GPU work %.1f ms, transfers %.1f ms, tiles %d x %d, %d tile convolutions)\n",
                        st.device.c_str(), st.seconds, st.fftSeconds * 1e3, st.transferSeconds * 1e3,
                        st.fftWidth, st.fftHeight, st.convolutions);
            // second frame: what the plugin costs per frame once the PSF spectra are on the GPU
            RenderStats st2;
            ImageRGB tmp;
            if ( gpu.render(in, tmp, &st2, &err) ) {
                std::printf("GPU next frame: %.3f s (GPU work %.1f ms, transfers %.1f ms, PSF spectra %s)\n",
                            st2.seconds, st2.fftSeconds * 1e3, st2.transferSeconds * 1e3,
                            st2.kernelCache ? "kept on the GPU" : "recomputed (not enough memory)");
            }
            rendered = true;
        } else {
            std::fprintf(stderr, "carender: GPU render failed (%s), using the CPU\n", err.c_str());
        }
    }
    if ( !rendered || compare ) {
        RenderStats sc;
        renderCpu(plan, in, outCpu, &sc);
        std::printf("CPU: %.3f s (light per wavelength %.2f s, convolutions %.2f s, tiles %d x %d, %d tile convolutions)\n",
                    sc.seconds, sc.sourceSeconds, sc.fftSeconds, sc.fftWidth, sc.fftHeight, sc.convolutions);
        if (rendered) {
            double md;
            const double db = psnr(out, outCpu, &md);
            std::printf("GPU vs CPU: max difference %.3g, PSNR %.1f dB\n", md, db);
        } else {
            out = outCpu;
        }
    }
    if (num(p, "exact", 0.) != 0.) {
        ImageRGB ex;
        const auto t0 = std::chrono::steady_clock::now();
        renderExactCpu(plan, in, ex);
        double md;
        const double db = psnr(out, ex, &md);
        std::printf("Exact convolution: %.1f s; difference to the render: max %.3g, PSNR %.1f dB\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), md, db);
    }
    if ( !saveExr(pos[1], out, alpha) ) {
        return 1;
    }
    std::printf("Wrote %s\n", pos[1].c_str());
    return 0;
}
