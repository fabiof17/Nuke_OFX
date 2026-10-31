/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX. GPL v2 or later, see LICENSE.
 * ***** END LICENSE BLOCK ***** */

/*
 * Engine tests, without an OFX host: they check the CPU engine against known
 * results, and, when a CUDA device is present, check that the GPU engine gives
 * the same image as the CPU engine for every filter, depth and option.
 *
 *   test_engine            run the tests
 *   test_engine --bench    also time a 2K render (CPU and GPU)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpCpuEngine.h"
#include "cpEngine.h"

using namespace cpg;

static int gFailures = 0;
static int gChecks = 0;

#define CHECK(cond, ...) \
    do { ++gChecks; if (!(cond)) { ++gFailures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Img
{
    int w, h, nc;
    PixelDepth depth;
    std::vector<unsigned char> bytes;
    size_t bpc() const { return depth == eDepthUByte ? 1 : (depth == eDepthUShort ? 2 : 4); }
    long long rowBytes() const { return (long long)(w * nc * bpc()); }
    Img(int w_, int h_, int nc_, PixelDepth d) : w(w_), h(h_), nc(nc_), depth(d), bytes( (size_t)w_ * h_ * nc_ * (d == eDepthUByte ? 1 : (d == eDepthUShort ? 2 : 4)), 0 ) {}
    float get(int x, int y, int c) const
    {
        size_t i = ( (size_t)y * w + x ) * nc + c;
        switch (depth) {
        case eDepthUByte: return bytes[i];
        case eDepthUShort: return ( (const unsigned short*)&bytes[0] )[i];
        default: return ( (const float*)&bytes[0] )[i];
        }
    }
    void set(int x, int y, int c, float v)
    {
        size_t i = ( (size_t)y * w + x ) * nc + c;
        switch (depth) {
        case eDepthUByte: bytes[i] = (unsigned char)v; break;
        case eDepthUShort: ( (unsigned short*)&bytes[0] )[i] = (unsigned short)v; break;
        default: ( (float*)&bytes[0] )[i] = v; break;
        }
    }
};

static void setMat(double* m, double a, double b, double c, double d, double e, double f, double g = 0, double h = 0, double i = 1)
{
    m[0] = a; m[1] = b; m[2] = c; m[3] = d; m[4] = e; m[5] = f; m[6] = g; m[7] = h; m[8] = i;
}

// a job whose source and destination cover (0,0)-(w,h)
static RenderJob makeJob(const Img& src, Img& dst, int filter)
{
    RenderJob job;
    job.nComp = src.nc;
    job.depth = src.depth;
    job.src.data = &src.bytes[0];
    job.src.rowBytes = src.rowBytes();
    job.src.x1 = 0; job.src.y1 = 0; job.src.x2 = src.w; job.src.y2 = src.h;
    job.mask.data = NULL;
    job.mask.rowBytes = 0;
    job.mask.x1 = job.mask.y1 = job.mask.x2 = job.mask.y2 = 0;
    job.dstData = &dst.bytes[0];
    job.dstRowBytes = dst.rowBytes();
    job.dstX1 = 0; job.dstY1 = 0;
    job.rw[0] = 0; job.rw[1] = 0; job.rw[2] = dst.w; job.rw[3] = dst.h;
    job.H.resize(9);
    setMat(&job.H[0], 1, 0, 0, 0, 1, 0);
    job.sp.filter = filter;
    job.sp.clamp = false;
    job.sp.blackOutside = true;
    job.motionblur = 0.;
    job.masked = false;
    job.doMask = false;
    job.maskInvert = false;
    job.mix = 1.;
    return job;
}

static void renderCPU(const RenderJob& job)
{
    CpuRenderer r(job);
    r.prepare();
    r.renderRows(job.rw[1], job.rw[3]);
}

static Img randomImage(int w, int h, int nc, PixelDepth d, unsigned seed)
{
    Img im(w, h, nc, d);
    std::srand(seed);
    float maxv = d == eDepthUByte ? 255.f : (d == eDepthUShort ? 65535.f : 1.f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < nc; ++c) {
                float v = (std::rand() / (float)RAND_MAX) * maxv;
                im.set(x, y, c, d == eDepthFloat ? v : std::floor(v));
            }
        }
    }
    return im;
}

static const char* filterName(int f)
{
    static const char* n[] = { "impulse", "box", "bilinear", "cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch", "ewagaussian", "ewalanczos3" };
    return (f >= 0 && f < eFilterCount) ? n[f] : "?";
}

static double maxAbsDiff(const Img& a, const Img& b, int x1, int y1, int x2, int y2)
{
    double m = 0;
    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            for (int c = 0; c < a.nc; ++c) {
                m = std::max(m, (double)std::fabs(a.get(x, y, c) - b.get(x, y, c)));
            }
        }
    }
    return m;
}

////////////////////////////////////////////////////////////////////////////////

static void testIdentity()
{
    Img src = randomImage(64, 48, 4, eDepthFloat, 1);
    // filters that interpolate (exact at pixel centers)
    const int exact[] = { eImpulse, eBilinear, eCubic, eKeys, eSimon, eRifman, eEWALanczos3, eBox };
    for (int f : exact) {
        Img dst(64, 48, 4, eDepthFloat);
        RenderJob job = makeJob(src, dst, f);
        renderCPU(job);
        double d = maxAbsDiff(src, dst, 0, 0, 64, 48);
        CHECK(d < 1e-4, "identity %s: max diff %g", filterName(f), d);
    }
}

static void testTranslation()
{
    Img src = randomImage(40, 30, 3, eDepthFloat, 2);
    for (int f = 0; f < eFilterCount; ++f) {
        if ( (f == eMitchell) || (f == eParzen) || (f == eNotch) || (f == eEWAGaussian) ) {
            continue; // smoothing filters
        }
        Img dst(40, 30, 3, eDepthFloat);
        RenderJob job = makeJob(src, dst, f);
        // dst(x,y) = src(x-3, y+2): inverse maps dst -> src
        setMat(&job.H[0], 1, 0, -3, 0, 1, 2);
        renderCPU(job);
        double m = 0;
        for (int y = 0; y < 28; ++y) {
            for (int x = 3; x < 40; ++x) {
                for (int c = 0; c < 3; ++c) {
                    m = std::max(m, (double)std::fabs( dst.get(x, y, c) - src.get(x - 3, y + 2, c) ));
                }
            }
        }
        CHECK(m < 1e-4, "translation %s: max diff %g", filterName(f), m);
        // outside is black
        CHECK(std::fabs(dst.get(0, 10, 0)) < 1e-4 && std::fabs(dst.get(1, 29, 1)) < 1e-4, "translation %s: black outside", filterName(f));
    }
}

// 1-pixel checkerboard minified 8x: a good filter gives a flat 0.5 grey
static void testAntiAliasing()
{
    const int W = 512;
    Img src(W, W, 1, eDepthFloat);
    for (int y = 0; y < W; ++y) {
        for (int x = 0; x < W; ++x) {
            src.set(x, y, 0, ( (x + y) & 1 ) ? 1.f : 0.f);
        }
    }
    struct { int f; double maxDev; } cases[] = {
        { eEWAGaussian, 0.02 }, { eEWALanczos3, 0.05 }, { eCubic, 0.05 }, { eBox, 0.02 }, { eBilinear, 0.05 }
    };
    for (auto cs : cases) {
        Img dst(64, 64, 1, eDepthFloat);
        RenderJob job = makeJob(src, dst, cs.f);
        // perspective minification: scale 8 plus a mild perspective term (scale grows to ~9.3 across the image)
        setMat(&job.H[0], 8, 0.3, 0, 0.2, 8, 0, 0.0002, 0.0001, 1);
        renderCPU(job);
        double dev = 0, mean = 0;
        int n = 0;
        for (int y = 4; y < 52; ++y) {
            for (int x = 4; x < 52; ++x) {
                dev = std::max(dev, std::fabs(dst.get(x, y, 0) - 0.5));
                mean += dst.get(x, y, 0);
                ++n;
            }
        }
        mean /= n;
        CHECK(dev < cs.maxDev, "anti-aliasing %s: max deviation from grey %g (mean %g)", filterName(cs.f), dev, mean);
    }
    // and impulse does alias (sanity check of the test itself)
    Img dst(64, 64, 1, eDepthFloat);
    RenderJob job = makeJob(src, dst, eImpulse);
    setMat(&job.H[0], 8, 0.3, 0, 0.2, 8, 0, 0.0002, 0.0001, 1);
    renderCPU(job);
    double dev = 0;
    for (int y = 4; y < 52; ++y) {
        for (int x = 4; x < 52; ++x) {
            dev = std::max(dev, std::fabs(dst.get(x, y, 0) - 0.5));
        }
    }
    CHECK(dev > 0.4, "impulse should alias, deviation %g", dev);
}

// horizontal motion over 10 pixels of a vertical edge gives a linear ramp
static void testMotionBlur()
{
    Img src(100, 8, 1, eDepthFloat);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 100; ++x) {
            src.set(x, y, 0, x < 50 ? 1.f : 0.f);
        }
    }
    Img dst(100, 8, 1, eDepthFloat);
    RenderJob job = makeJob(src, dst, eBilinear);
    job.sp.blackOutside = false;
    const int count = 1000;
    job.H.resize(9 * count);
    for (int t = 0; t < count; ++t) {
        double tx = 10.0 * t / (count - 1);
        setMat(&job.H[9 * t], 1, 0, -tx, 0, 1, 0);
    }
    job.motionblur = 4.; // high quality
    renderCPU(job);
    double m = 0;
    for (int x = 40; x < 70; ++x) {
        // expected: fraction of the shutter during which src(x - tx + 0.5) is 1, i.e. x+0.5-tx < 50
        double xc = x + 0.5;
        double frac = (xc - 50.0) / 10.0; // fraction of time the edge has passed
        double expected = 1.0 - std::min(1.0, std::max(0.0, frac));
        // bilinear softens by half a pixel: compare loosely
        m = std::max(m, std::fabs(dst.get(x, 4, 0) - expected));
    }
    CHECK(m < 0.08, "motion blur ramp: max error %g", m);
}

static void testMaskMix()
{
    Img src = randomImage(20, 20, 4, eDepthFloat, 3);
    Img mask(20, 20, 1, eDepthFloat);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            mask.set(x, y, 0, x < 10 ? 0.f : 1.f);
        }
    }
    Img dst(20, 20, 4, eDepthFloat);
    RenderJob job = makeJob(src, dst, eImpulse);
    setMat(&job.H[0], 1, 0, 0, 0, 1, -1); // shift by one row
    job.masked = true;
    job.doMask = true;
    job.mask.data = &mask.bytes[0];
    job.mask.rowBytes = mask.rowBytes();
    job.mask.x1 = 0; job.mask.y1 = 0; job.mask.x2 = 20; job.mask.y2 = 20;
    job.mix = 0.5;
    renderCPU(job);
    double m = 0;
    for (int y = 1; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            for (int c = 0; c < 4; ++c) {
                float a = x < 10 ? 0.f : 0.5f;
                float expected = src.get(x, y - 1, c) * a + (1 - a) * src.get(x, y, c);
                m = std::max(m, (double)std::fabs(dst.get(x, y, c) - expected));
            }
        }
    }
    CHECK(m < 1e-5, "mask/mix: max diff %g", m);
}

static void testIntegerDepths()
{
    const PixelDepth depths[] = { eDepthUByte, eDepthUShort };
    for (PixelDepth d : depths) {
        Img src = randomImage(33, 17, 3, d, 4);
        Img dst(33, 17, 3, d);
        RenderJob job = makeJob(src, dst, eKeys);
        renderCPU(job);
        double diff = maxAbsDiff(src, dst, 0, 0, 33, 17);
        CHECK(diff == 0, "identity depth %d: max diff %g", (int)d, diff);
    }
}

// render window and source bounds that do not start at 0 (tiles)
static void testOffsetsAndTiles()
{
    Img src = randomImage(50, 40, 2, eDepthFloat, 5);
    Img full(50, 40, 2, eDepthFloat), tiled(50, 40, 2, eDepthFloat);
    RenderJob job = makeJob(src, full, eEWALanczos3);
    setMat(&job.H[0], 1.3, 0.2, -4, -0.1, 1.1, 3, 0.001, 0.0005, 1);
    renderCPU(job);
    for (int ty = 0; ty < 40; ty += 16) {
        for (int tx = 0; tx < 50; tx += 16) {
            RenderJob t = job;
            t.dstData = &tiled.bytes[0];
            t.rw[0] = tx; t.rw[1] = ty; t.rw[2] = std::min(tx + 16, 50); t.rw[3] = std::min(ty + 16, 40);
            renderCPU(t);
        }
    }
    double d = maxAbsDiff(full, tiled, 0, 0, 50, 40);
    CHECK(d == 0, "tiled render differs: %g", d);
}

////////////////////////////////////////////////////////////////////////////////
// GPU vs CPU

static void compareGPU(const RenderJob& cpuJob, const Img& cpuDst, Img& gpuDst, int device, const char* what, double tol)
{
    RenderJob g = cpuJob;
    g.dstData = &gpuDst.bytes[0];
    std::string err;
    bool ok = cudaRender(g, device, &err);
    CHECK(ok, "%s: GPU render failed: %s", what, err.c_str());
    if (!ok) {
        return;
    }
    double d = maxAbsDiff(cpuDst, gpuDst, cpuJob.rw[0], cpuJob.rw[1], cpuJob.rw[2], cpuJob.rw[3]);
    CHECK(d <= tol, "%s: GPU/CPU max diff %g (tolerance %g)", what, d, tol);
}

static void testGPU(int device)
{
    const PixelDepth depths[] = { eDepthFloat, eDepthUShort, eDepthUByte };
    for (PixelDepth d : depths) {
        for (int nc = 1; nc <= 4; ++nc) {
            Img src = randomImage(96, 80, nc, d, 10 + nc);
            float maxv = d == eDepthUByte ? 255.f : (d == eDepthUShort ? 65535.f : 1.f);
            for (int f = 0; f < eFilterCount; ++f) {
                for (int mb = 0; mb < 2; ++mb) {
                    Img c(96, 80, nc, d), g(96, 80, nc, d);
                    RenderJob job = makeJob(src, c, f);
                    job.sp.clamp = (f % 2) == 0;
                    job.sp.blackOutside = (f % 3) != 0;
                    if (mb) {
                        const int count = 200;
                        job.H.resize(9 * count);
                        for (int t = 0; t < count; ++t) {
                            double a = 0.2 * t / count;
                            setMat(&job.H[9 * t], 2.0 * std::cos(a), -std::sin(a), 5 + t * 0.05, std::sin(a), 1.5 * std::cos(a), -3, 0.002, 0.001, 1);
                        }
                        job.motionblur = 1.;
                    } else {
                        setMat(&job.H[0], 2.0, 0.3, 5, -0.2, 1.5, -3, 0.002, 0.001, 1);
                    }
                    job.rw[0] = 5; job.rw[1] = 7; job.rw[2] = 90; job.rw[3] = 77;
                    renderCPU(job);
                    char what[128];
                    std::snprintf(what, sizeof(what), "gpu depth=%d nc=%d %s mb=%d", (int)d, nc, filterName(f), mb);
                    // float rounding differences (fast math, FMA): a few ULPs on most pixels,
                    // more where adaptive motion blur takes a different number of samples
                    compareGPU(job, c, g, device, what, (mb ? 0.03 : 0.002) * maxv + (d == eDepthFloat ? 0 : 1));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

static void bench(int device, bool hasGPU)
{
    const int W = 2048, Hh = 1080;
    Img src = randomImage(W, Hh, 4, eDepthFloat, 42);
    Img dst(W, Hh, 4, eDepthFloat);
    const int filters[] = { eCubic, eEWALanczos3 };
    for (int f : filters) {
        for (int mb = 0; mb < 2; ++mb) {
            RenderJob job = makeJob(src, dst, f);
            if (mb) {
                const int count = 1000;
                job.H.resize(9 * count);
                for (int t = 0; t < count; ++t) {
                    setMat(&job.H[9 * t], 1.5, 0.2, -100 + t * 0.02, -0.1, 1.4, 50, 0.0002, 0.0001, 1);
                }
                job.motionblur = 1.;
            } else {
                setMat(&job.H[0], 1.5, 0.2, -100, -0.1, 1.4, 50, 0.0002, 0.0001, 1);
            }
            auto t0 = std::chrono::steady_clock::now();
            renderCPU(job); // single thread here; the plugin uses all cores
            auto t1 = std::chrono::steady_clock::now();
            double cpu = std::chrono::duration<double>(t1 - t0).count();
            std::printf("bench 2048x1080 RGBA float %-12s mb=%d  CPU (1 thread): %8.1f ms", filterName(f), mb, cpu * 1000);
            if (hasGPU) {
                std::string err;
                cudaRender(job, device, &err); // warm-up
                t0 = std::chrono::steady_clock::now();
                cudaRender(job, device, &err);
                t1 = std::chrono::steady_clock::now();
                std::printf("   GPU (incl. transfers): %8.1f ms", std::chrono::duration<double>(t1 - t0).count() * 1000);
            }
            std::printf("\n");
        }
    }
}

int main(int argc, char** argv)
{
    bool doBench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    testIdentity();
    testTranslation();
    testAntiAliasing();
    testMotionBlur();
    testMaskMix();
    testIntegerDepths();
    testOffsetsAndTiles();

    CudaDeviceInfo info = cudaQueryDevice(0);
    std::printf("%s\n", info.message.c_str());
    if (info.available) {
        testGPU(info.device);
    } else {
        std::printf("no CUDA device: GPU tests skipped\n");
    }
    if (doBench) {
        bench(info.device, info.available);
    }
    std::printf("%d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
}
