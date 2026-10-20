// MasterKeyerCUDA.cu
// CUDA back-end of MasterKeyer_OFX: same stages and arithmetic as the CPU back-end (built
// without fast-math and without FMA contraction). The host images are streamed straight into
// page-locked transfer memory; device buffers are cached between renders (mutex protected).
#include "MasterKeyerCore.h"
#include "PinnedStaging.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mk {
namespace {

using namespace mkshared;

#define MK_CUDA_CHECK(call)                                                          \
    do {                                                                             \
        cudaError_t e__ = (call);                                                    \
        if (e__ != cudaSuccess) { error = std::string(#call) + " : " + cudaGetErrorString(e__); return false; } \
    } while (0)

struct DevBuf {
    void*  ptr = nullptr;
    size_t size = 0;
    bool ensure(size_t bytes) {
        if (bytes <= size) return true;
        release();
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; return false; }
        size = bytes;
        return true;
    }
    void release() { if (ptr) cudaFree(ptr); ptr = nullptr; size = 0; }
    template <class T> T* as() const { return static_cast<T*>(ptr); }
    float* f() const { return static_cast<float*>(ptr); }
};

struct DevCache {
    std::mutex mutex;
    DevBuf src, bg, keyIn, garbage, holdout, out;
    DevBuf ycc, ycc2, screen, alpha, tmp1, dist, pyramid, bgSmall, tmpSmall;
    DevBuf nearestY, edtV, edtZ, edtD, seedT, seeds;
    PinnedStaging staging;
    cudaStream_t stream = nullptr;
    void release() {
        DevBuf* all[] = { &src, &bg, &keyIn, &garbage, &holdout, &out, &ycc, &ycc2, &screen, &alpha, &tmp1, &dist, &pyramid,
                          &bgSmall, &tmpSmall, &nearestY, &edtV, &edtZ, &edtD, &seedT, &seeds };
        for (DevBuf* b : all) b->release();
        staging.release();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};
DevCache& cache() { static DevCache* c = new DevCache(); return *c; }   // leaked on purpose (DLL unload order)

struct StageProfiler {
    bool on = false;
    cudaStream_t st = nullptr;
    std::vector<std::pair<std::string, cudaEvent_t>> marks;
    std::string last;
    void begin(cudaStream_t s) { if (!on) return; st = s; clear(); mark("start"); }
    void mark(const std::string& name)
    {
        if (!on) return;
        cudaEvent_t e;
        if (cudaEventCreate(&e) != cudaSuccess) return;
        cudaEventRecord(e, st);
        marks.push_back({ name, e });
    }
    void end()
    {
        if (!on || marks.empty()) return;
        std::string r;
        float total = 0.f;
        for (size_t i = 1; i < marks.size(); ++i) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, marks[i - 1].second, marks[i].second);
            total += ms;
            char buf[160];
            std::snprintf(buf, sizeof buf, "      %-22s %7.2f ms\n", marks[i].first.c_str(), ms);
            r += buf;
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "      %-22s %7.2f ms\n", "= GPU total", total);
        last = r + buf;
        clear();
    }
    void clear() { for (auto& m : marks) cudaEventDestroy(m.second); marks.clear(); }
};
StageProfiler& profiler() { static StageProfiler* p = new StageProfiler(); return *p; }

inline dim3 grid2d(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

#define MK_PIXEL                                                                                        \
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;     \
    if (x >= p.width || y >= p.height) return;                                                          \
    const size_t i = (size_t)y * p.width + x;

// ------------------------------------------------------------------- kernels
__global__ void kToYCC(Params p, const float* src3, float* ycc)
{
    MK_PIXEL
    toYCC(src3 + 3 * i, ycc + 4 * i);
}

// ycc is replaced by the degrained RGB (alpha 0); each thread only touches its own pixel.
__global__ void kFromYCC(Params p, float* ycc, const float* yccBlur)
{
    MK_PIXEL
    const float in[4] = { ycc[4 * i], ycc[4 * i + 1], ycc[4 * i + 2], ycc[4 * i + 3] };
    float* o = ycc + 4 * i;
    fromYCC(in, yccBlur + 4 * i, p.grainEdges, o);
    o[3] = 0.f;
}

// mc: colour used for the matte, mcStride floats per pixel (3 = plate RGB, 4 = degrained).
__global__ void kSeed(Params p, const float* mc, int mcStride, float* base)
{
    MK_PIXEL
    screenSeed(p, mc + (size_t)mcStride * i, base + 4 * i);
}

__global__ void kMatte(Params p, const float* mc, int mcStride, const float* pushed, float* screen, float* alpha)
{
    MK_PIXEL
    float* S = screen + 4 * i;
    screenAt(p, pushed ? pushed + 4 * i : nullptr, S);
    S[3] = 0.f;
    alpha[i] = mattePixel(p, mc + (size_t)mcStride * i, S);
}

__global__ void kMinMax(const float* in, float* out, int w, int h, int r, int dilate, int alongRows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float m = alongRows ? in[(size_t)y * w + diClampi(x - r, 0, w - 1)] : in[(size_t)diClampi(y - r, 0, h - 1) * w + x];
    for (int d = -r + 1; d <= r; ++d) {
        const float v = alongRows ? in[(size_t)y * w + diClampi(x + d, 0, w - 1)] : in[(size_t)diClampi(y + d, 0, h - 1) * w + x];
        m = dilate ? (v > m ? v : m) : (v < m ? v : m);
    }
    out[(size_t)y * w + x] = m;
}

// Garbage / holdout, and the screen mask used by the distance transform.
__global__ void kGarbage(Params p, float* alpha, const float* garbage, const float* holdout, float* screenMask)
{
    MK_PIXEL
    const float a = garbageHoldout(p, alpha[i], garbage ? garbage + i : nullptr, holdout ? holdout + i : nullptr);
    alpha[i] = a;
    if (screenMask) screenMask[i] = a <= 0.02f ? 1.f : 0.f;
}

__global__ void kPull(const float* fine, int fw, int fh, float* coarse, int cw, int ch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || y >= ch) return;
    pullPixel(fine, fw, fh, x, y, coarse + 4 * ((size_t)y * cw + x));
}

__global__ void kPush(float* fine, int fw, int fh, const float* coarse, int cw, int ch)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fw || y >= fh) return;
    pushPixel(fine + 4 * ((size_t)y * fw + x), coarse, cw, ch, x, y);
}

__global__ void kEdtColumns(const float* mask, int w, int h, int* nearestY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    edtColumn(mask, w, h, x, nearestY);
}

__global__ void kEdtRows(const int* nyT, int w, int h, int* v, long long* zn, int* zd, Seed* seedsT)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    edtRow(nyT + y, w, y, (size_t)h, v + y, zn + y, zd + y, seedsT + y);
}

__global__ void kDistance(const Seed* seeds, float* dist, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dist[(size_t)y * w + x] = seedDistance(seeds[(size_t)y * w + x], x, y);
}

__global__ void kDownsample(const float* in, int w, int h, int nch, int f, float* out, int ws, int hs)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= ws || Y >= hs) return;
    downsamplePixel(in, w, h, nch, f, X, Y, out + (size_t)nch * ((size_t)Y * ws + X));
}

struct F3 { float x, y, z; };

template <class T>
__global__ void kTranspose(const T* in, T* out, int w, int h)
{
    __shared__ T tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int x = bx + threadIdx.x, y = by + j;
        if (x < w && y < h) tile[j][threadIdx.x] = in[(size_t)y * w + x];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int x = by + threadIdx.x, y = bx + j;
        if (x < h && y < w) out[(size_t)y * h + x] = tile[threadIdx.x][j];
    }
}

// Sliding box along columns, in segments (see push_IT_OFX). Channels >= nch copied through.
__global__ void kSlideSegments(const float* in, float* out, int w, int h, int pstride, int nch, int r, int seg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, sg = blockIdx.y;
    if (x >= w) return;
    const int y0 = sg * seg, y1 = y0 + seg < h ? y0 + seg : h;
    if (y0 >= h) return;
    const size_t stride = (size_t)w * pstride;
    const float* a = in + (size_t)x * pstride;
    float* o = out + (size_t)x * pstride;
    const double norm = 1.0 / (double)(2 * r + 1);
    double sum[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = y0 - r; i <= y0 + r; ++i) {
        const float* q = a + (size_t)diClampi(i, 0, h - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += q[c];
    }
    for (int y = y0; y < y1; ++y) {
        const float* q = a + (size_t)y * stride;
        float* d = o + (size_t)y * stride;
        for (int c = 0; c < nch; ++c) d[c] = (float)(sum[c] * norm);
        for (int c = nch; c < pstride; ++c) d[c] = q[c];
        const float* qa = a + (size_t)diClampi(y + r + 1, 0, h - 1) * stride;
        const float* qs = a + (size_t)diClampi(y - r, 0, h - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += (double)qa[c] - (double)qs[c];
    }
}

__global__ void kFinal(Params p, const float* src3, const float* screen, const float* alpha, const float* dist,
                       const float* bg, const float* bgSmall, int ws, int hs, int f, float* out)
{
    MK_PIXEL
    float b3[3];
    if (bgSmall) sampleReduced(bgSmall, ws, hs, 3, f, x, y, b3);
    finalPixel(p, src3 + 3 * i, screen + 4 * i, alpha[i], dist ? dist[i] : 1e30f, bg ? bg + 3 * i : nullptr,
               bgSmall ? b3 : nullptr, out + 4 * i);
}

// ------------------------------------------------------------------- helpers
void transpose(const float* in, float* out, int w, int h, int pstride, cudaStream_t st)
{
    const dim3 tb(32, 8), tg((w + 31) / 32, (h + 31) / 32);
    if (pstride == 4)      kTranspose<float4><<<tg, tb, 0, st>>>((const float4*)in, (float4*)out, w, h);
    else if (pstride == 3) kTranspose<F3><<<tg, tb, 0, st>>>((const F3*)in, (F3*)out, w, h);
    else                   kTranspose<float><<<tg, tb, 0, st>>>(in, out, w, h);
}

void slideColumns3(float* a, float* b, int w, int h, int pstride, int nch, int r, cudaStream_t st)
{
    const int seg = (2 * r + 1) > 48 ? (2 * r + 1) : 48;
    const unsigned tb = 128;
    const dim3 g((w + tb - 1) / tb, (h + seg - 1) / seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, pstride, nch, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(b, a, w, h, pstride, nch, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, pstride, nch, r, seg);
}

// 3 box passes along the rows then the columns, in place on img (tmp: scratch of the same size).
void blur(float* img, float* tmp, int w, int h, int pstride, int nch, int r, cudaStream_t st)
{
    transpose(img, tmp, w, h, pstride, st);
    slideColumns3(tmp, img, h, w, pstride, nch, r, st);
    transpose(img, tmp, h, w, pstride, st);
    slideColumns3(tmp, img, w, h, pstride, nch, r, st);
}

void pushPull(float* pyr, const std::vector<int>& lw, const std::vector<int>& lh, const std::vector<size_t>& loff,
              cudaStream_t st)
{
    const dim3 b(16, 16);
    for (size_t l = 1; l < lw.size(); ++l)
        kPull<<<grid2d(lw[l], lh[l], b), b, 0, st>>>(pyr + loff[l - 1], lw[l - 1], lh[l - 1], pyr + loff[l], lw[l], lh[l]);
    for (int l = (int)lw.size() - 2; l >= 0; --l)
        kPush<<<grid2d(lw[l], lh[l], b), b, 0, st>>>(pyr + loff[l], lw[l], lh[l], pyr + loff[l + 1], lw[l + 1], lh[l + 1]);
}

} // namespace

// =================================================================== public
bool cudaIsCompiledIn() { return true; }

bool cudaDeviceInfo(std::string& description)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        cudaGetLastError();
        description = "CUDA: no NVIDIA GPU available (CPU rendering)";
        return false;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        description = "CUDA: cannot read the GPU (CPU rendering)";
        return false;
    }
    description = std::string("Local GPU: ") + prop.name + " (" + std::to_string(prop.totalGlobalMem >> 20) + " MB)";
    return true;
}

void cudaReleaseResources()
{
    std::lock_guard<std::mutex> lock(cache().mutex);
    cache().release();
}

void cudaSetProfiling(bool on) { profiler().on = on; }
std::string cudaLastProfile() { return profiler().last; }

bool processCUDA(const Params& p, const Inputs& in, float* out, std::string& error)
{
    const size_t w = (size_t)p.width;
    auto reader = [w](const float* img, int nch) {
        return RowReader([img, nch, w](int y0, int y1, float* o) {
            std::memcpy(o, img + nch * w * y0, nch * w * (y1 - y0) * sizeof(float));
        });
    };
    const RowReader rs = [&](int y0, int y1, float* o) {          // RGBA -> RGB
        for (size_t k = (size_t)w * y0; k < (size_t)w * y1; ++k, o += 3) {
            o[0] = in.src[4 * k]; o[1] = in.src[4 * k + 1]; o[2] = in.src[4 * k + 2];
        }
    };
    const RowReader rk = reader(in.keyIn, 3);
    const RowReader rb = reader(in.bg, 3), rg = reader(in.garbage, 1), rh = reader(in.holdout, 1);
    RowInputs ri;
    ri.src = &rs;
    ri.bg = in.bg ? &rb : nullptr;
    ri.keyIn = in.keyIn ? &rk : nullptr;
    ri.garbage = in.garbage ? &rg : nullptr;
    ri.holdout = in.holdout ? &rh : nullptr;
    const RowWriter wo = [&](int y0, int y1, const float* i) { std::memcpy(out + 4 * w * y0, i, 4 * w * (y1 - y0) * sizeof(float)); };
    return processCUDARows(p, ri, wo, error);
}

bool processCUDARows(const Params& p, const RowInputs& in, const RowWriter& out, std::string& error)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0 || !in.src) return true;
    const size_t N = (size_t)w * h;
    const size_t f1 = N * sizeof(float), f4 = 4 * f1;

    const bool local = p.screenSource == kScreenLocal;
    const bool dist = needsDistance(p);
    const bool bgBlur = needsBGBlur(p) && in.bg;

    std::vector<int> lw(1, w), lh(1, h);
    std::vector<size_t> loff(1, 0);
    size_t pyr = f4;
    while (lw.back() > 1 || lh.back() > 1) {
        lw.push_back((lw.back() + 1) / 2);
        lh.push_back((lh.back() + 1) / 2);
        loff.push_back(pyr / sizeof(float));
        pyr += (size_t)4 * lw.back() * lh.back() * sizeof(float);
    }

    DevCache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.stream) MK_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaStream_t st = c.stream;

    bool ok = c.src.ensure(3 * f1) && c.out.ensure(f4) && c.screen.ensure(f4) && c.alpha.ensure(f1) && c.tmp1.ensure(f1) &&
              (!in.bg || c.bg.ensure(3 * f1)) && (!in.keyIn || c.keyIn.ensure(3 * f1)) && (!in.garbage || c.garbage.ensure(f1)) && (!in.holdout || c.holdout.ensure(f1));
    if (ok && p.degrain) ok = c.ycc.ensure(f4) && c.ycc2.ensure(f4);
    if (ok && local) ok = c.pyramid.ensure(pyr);
    if (ok && dist)
        ok = c.dist.ensure(f1) && c.nearestY.ensure(2 * N * sizeof(int)) && c.edtV.ensure(N * sizeof(int)) &&
             c.edtZ.ensure(N * sizeof(long long)) && c.edtD.ensure(N * sizeof(int)) &&
             c.seedT.ensure(N * sizeof(Seed)) && c.seeds.ensure(N * sizeof(Seed));
    int rf, br;
    bgReducePlan(p.bgBlur, &rf, &br);
    const int ws = (w + rf - 1) / rf, hs = (h + rf - 1) / rf;
    const size_t small = (size_t)ws * hs * sizeof(float);
    if (ok && bgBlur) ok = c.bgSmall.ensure(3 * small) && c.tmpSmall.ensure(3 * small);
    if (!ok) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }

    auto asFill = [](const RowReader& r) {
        return [&r](int y0, int y1, void* d) { r(y0, y1, static_cast<float*>(d)); };
    };
    profiler().begin(st);
    MK_CUDA_CHECK(c.staging.uploadRows(c.src.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(*in.src)));
    if (in.keyIn)   MK_CUDA_CHECK(c.staging.uploadRows(c.keyIn.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(*in.keyIn)));
    if (in.bg)      MK_CUDA_CHECK(c.staging.uploadRows(c.bg.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(*in.bg)));
    if (in.garbage) MK_CUDA_CHECK(c.staging.uploadRows(c.garbage.f(), (size_t)w * sizeof(float), h, st, asFill(*in.garbage)));
    if (in.holdout) MK_CUDA_CHECK(c.staging.uploadRows(c.holdout.f(), (size_t)w * sizeof(float), h, st, asFill(*in.holdout)));
    profiler().mark("upload (host->GPU)");

    const dim3 b(16, 16), g = grid2d(w, h, b);

    // 1. colour used for the matte
    const float* mc = in.keyIn ? c.keyIn.f() : c.src.f();     // Key In or plate, RGB
    int mcStride = 3;
    if (p.degrain) {
        kToYCC<<<g, b, 0, st>>>(p, mc, c.ycc.f());
        MK_CUDA_CHECK(cudaMemcpyAsync(c.ycc2.f(), c.ycc.f(), f4, cudaMemcpyDeviceToDevice, st));
        const int r = boxRadiusForBlur(grainBlurSize(p));
        if (r > 0) blur(c.ycc2.f(), c.out.f(), w, h, 4, degrainChannels(p), r, st);
        kFromYCC<<<g, b, 0, st>>>(p, c.ycc.f(), c.ycc2.f());
        mc = c.ycc.f();
        mcStride = 4;
        profiler().mark("degrain");
    }

    // 2. local screen
    if (local) {
        kSeed<<<g, b, 0, st>>>(p, mc, mcStride, c.pyramid.f());
        pushPull(c.pyramid.f(), lw, lh, loff, st);
        profiler().mark("local screen");
    }

    // 3. matte
    kMatte<<<g, b, 0, st>>>(p, mc, mcStride, local ? c.pyramid.f() : nullptr, c.screen.f(), c.alpha.f());
    if (p.shrink != 0) {
        const int r = std::abs(p.shrink), dil = p.shrink > 0 ? 1 : 0;
        kMinMax<<<g, b, 0, st>>>(c.alpha.f(), c.tmp1.f(), w, h, r, dil, 1);
        kMinMax<<<g, b, 0, st>>>(c.tmp1.f(), c.alpha.f(), w, h, r, dil, 0);
    }
    const int mr = boxRadiusForBlur(p.matteBlur);
    if (mr > 0) blur(c.alpha.f(), c.tmp1.f(), w, h, 1, 1, mr, st);
    kGarbage<<<g, b, 0, st>>>(p, c.alpha.f(), in.garbage ? c.garbage.f() : nullptr, in.holdout ? c.holdout.f() : nullptr,
                              dist ? c.tmp1.f() : nullptr);
    profiler().mark("matte");

    // 4. distance to the screen
    if (dist) {
        int* ny = c.nearestY.as<int>();
        int* nyT = ny + N;
        const unsigned tb = 64;
        const dim3 tt(32, 8);
        kEdtColumns<<<(w + tb - 1) / tb, tb, 0, st>>>(c.tmp1.f(), w, h, ny);
        kTranspose<int><<<dim3((w + 31) / 32, (h + 31) / 32), tt, 0, st>>>(ny, nyT, w, h);
        kEdtRows<<<(h + tb - 1) / tb, tb, 0, st>>>(nyT, w, h, c.edtV.as<int>(), c.edtZ.as<long long>(), c.edtD.as<int>(),
                                                     c.seedT.as<Seed>());
        kTranspose<Seed><<<dim3((h + 31) / 32, (w + 31) / 32), tt, 0, st>>>(c.seedT.as<Seed>(), c.seeds.as<Seed>(), h, w);
        kDistance<<<g, b, 0, st>>>(c.seeds.as<Seed>(), c.dist.f(), w, h);
        profiler().mark("distance to screen");
    }

    // 5. blurred background (reduced image)
    if (bgBlur) {
        kDownsample<<<grid2d(ws, hs, b), b, 0, st>>>(c.bg.f(), w, h, 3, rf, c.bgSmall.f(), ws, hs);
        if (br > 0) blur(c.bgSmall.f(), c.tmpSmall.f(), ws, hs, 3, 3, br, st);
        profiler().mark("background blur");
    }

    // 6. despill, blend, output
    kFinal<<<g, b, 0, st>>>(p, c.src.f(), c.screen.f(), c.alpha.f(), dist ? c.dist.f() : nullptr,
                            in.bg ? c.bg.f() : nullptr, bgBlur ? c.bgSmall.f() : nullptr, ws, hs, rf, c.out.f());
    MK_CUDA_CHECK(cudaGetLastError());
    profiler().mark("spill + blend");
    MK_CUDA_CHECK(c.staging.downloadRows(c.out.f(), 4 * (size_t)w * sizeof(float), h, st,
                                         [&](int y0, int y1, const void* i) { out(y0, y1, static_cast<const float*>(i)); }));
    profiler().mark("download (GPU->host)");
    MK_CUDA_CHECK(cudaStreamSynchronize(st));
    profiler().end();
    return true;
}

} // namespace mk
