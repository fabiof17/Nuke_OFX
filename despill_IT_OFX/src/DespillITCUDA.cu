// DespillITCUDA.cu
// CUDA back-end of despill_IT_OFX: same stages and arithmetic as the CPU back-end (built
// without fast-math and without FMA contraction). The host images are streamed straight into
// page-locked transfer memory; device buffers are cached between renders (mutex protected).
#include "DespillITCore.h"
#include "PinnedStaging.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace dspit {
namespace {

#define DI_CUDA_CHECK(call)                                                          \
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
    DevBuf src, key, clean, bg, protect, mask, out;
    DevBuf screen, alpha, screenMask, dist, pyramid, core, bgSmall, alphaSmall, tmpSmall;
    DevBuf nearestY, edtV, edtZ, edtD, seedT, seeds;
    PinnedStaging staging;
    cudaStream_t stream = nullptr;
    void release() {
        DevBuf* all[] = { &src, &key, &clean, &bg, &protect, &mask, &out, &screen, &alpha, &screenMask, &dist,
                          &pyramid, &core, &bgSmall, &alphaSmall, &tmpSmall, &nearestY, &edtV, &edtZ, &edtD, &seedT, &seeds };
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

// ------------------------------------------------------------------- kernels
__global__ void kPass1(Params p, const float* src, const float* key, float* screenBase)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    keyPass1(p, src + 4 * i, key ? key + i : nullptr, screenBase + 4 * i);
}

// pushed: local screen pyramid level 0 (or null). core is written in place of it when
// they share the buffer (each thread reads then writes its own pixel only).
__global__ void kPass2(Params p, const float* src, const float* key, const float* clean, const float* protect,
                       const float* pushed, float* screen, float* alpha, float* screenMask, float* core)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    float S[4];
    localScreen(p, pushed ? pushed + 4 * i : nullptr, clean ? clean + 3 * i : nullptr, S);
    S[3] = 0.f;
    const float pr = protection(p, src + 4 * i, protect ? protect + i : nullptr);
    float c4[4], m;
    const float a = keyPass2(p, src + 4 * i, key ? key + i : nullptr, S, pr, c4, &m);
    float* so = screen + 4 * i;
    so[0] = S[0]; so[1] = S[1]; so[2] = S[2]; so[3] = 0.f;
    alpha[i] = a;
    screenMask[i] = m;
    float* co = core + 4 * i;
    co[0] = c4[0]; co[1] = c4[1]; co[2] = c4[2]; co[3] = c4[3];
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

// bgSmall / alphaSmall: blurred BG and key on the reduced image (ws x hs, factor f), or null.
__global__ void kFinal(Params p, const float* src, const float* screen, const float* alpha, const float* dist,
                       const float* core, const float* bgSmall, const float* alphaSmall, int ws, int hs, int f,
                       const float* protect, const float* mask, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    float bg3[3], a1 = 1.f;
    if (bgSmall) sampleReduced(bgSmall, ws, hs, 3, f, x, y, bg3);
    if (alphaSmall) sampleReduced(alphaSmall, ws, hs, 1, f, x, y, &a1);
    finalPixel(p, src + 4 * i, screen + 4 * i, alpha[i], dist ? dist[i] : 0.f, core ? core + 4 * i : nullptr,
               bgSmall ? bg3 : nullptr, 1.f - a1, protect ? protect + i : nullptr, mask ? mask[i] : 1.f, out + 4 * i);
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

// Push-pull in place on the RGBW image at the start of `pyr` (levels follow it).
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
    const RowReader rs = reader(in.src, 4), rk = reader(in.key, 1), rc = reader(in.clean, 3), rb = reader(in.bg, 3),
                    rp = reader(in.protect, 1), rm = reader(in.mask, 1);
    RowInputs ri;
    ri.src = &rs;
    ri.key = in.key ? &rk : nullptr;
    ri.clean = in.clean ? &rc : nullptr;
    ri.bg = in.bg ? &rb : nullptr;
    ri.protect = in.protect ? &rp : nullptr;
    ri.mask = in.mask ? &rm : nullptr;
    const RowWriter wo = [&](int y0, int y1, const float* i) { std::memcpy(out + 4 * w * y0, i, 4 * w * (y1 - y0) * sizeof(float)); };
    return processCUDARows(p, ri, wo, error);
}

bool processCUDARows(const Params& p, const RowInputs& in, const RowWriter& out, std::string& error)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0 || !in.src) return true;
    const size_t N = (size_t)w * h;
    const size_t f1 = N * sizeof(float), f4 = 4 * f1;

    // pyramid layout (level 0 = full image)
    std::vector<int> lw(1, w), lh(1, h);
    std::vector<size_t> loff(1, 0);
    size_t pyr = f4;
    while (lw.back() > 1 || lh.back() > 1) {
        lw.push_back((lw.back() + 1) / 2);
        lh.push_back((lh.back() + 1) / 2);
        loff.push_back(pyr / sizeof(float));
        pyr += (size_t)4 * lw.back() * lh.back() * sizeof(float);
    }

    const bool local = needsLocalScreen(p), core = needsCore(p);
    const bool bgBlur = needsBGBlur(p) && in.bg, wrap = needsWrap(p) && in.bg;

    DevCache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.stream) DI_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaStream_t st = c.stream;

    bool ok = c.src.ensure(f4) && c.out.ensure(f4) && c.screen.ensure(f4) && c.alpha.ensure(f1) &&
              c.screenMask.ensure(f1) && c.pyramid.ensure(pyr) &&
              (!in.key || c.key.ensure(f1)) && (!in.clean || c.clean.ensure(3 * f1)) && (!in.bg || c.bg.ensure(3 * f1)) &&
              (!in.protect || c.protect.ensure(f1)) && (!in.mask || c.mask.ensure(f1));
    if (ok && p.spatial)
        ok = c.dist.ensure(f1) && c.nearestY.ensure(2 * N * sizeof(int)) && c.edtV.ensure(N * sizeof(int)) &&
             c.edtZ.ensure(N * sizeof(long long)) && c.edtD.ensure(N * sizeof(int)) &&
             c.seedT.ensure(N * sizeof(Seed)) && c.seeds.ensure(N * sizeof(Seed));
    int rf, br;
    bgReducePlan(p.bgBlur, &rf, &br);
    const int ws = (w + rf - 1) / rf, hs = (h + rf - 1) / rf;
    const size_t small = (size_t)ws * hs * sizeof(float);
    if (ok && bgBlur) ok = c.bgSmall.ensure(3 * small) && c.tmpSmall.ensure(3 * small);
    if (ok && wrap) ok = c.alphaSmall.ensure(small) && c.tmpSmall.ensure(3 * small);
    if (!ok) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }

    auto asFill = [](const RowReader& r) {
        return [&r](int y0, int y1, void* d) { r(y0, y1, static_cast<float*>(d)); };
    };
    profiler().begin(st);
    DI_CUDA_CHECK(c.staging.uploadRows(c.src.f(), 4 * (size_t)w * sizeof(float), h, st, asFill(*in.src)));
    if (in.key)     DI_CUDA_CHECK(c.staging.uploadRows(c.key.f(), (size_t)w * sizeof(float), h, st, asFill(*in.key)));
    if (in.clean)   DI_CUDA_CHECK(c.staging.uploadRows(c.clean.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(*in.clean)));
    if (in.bg)      DI_CUDA_CHECK(c.staging.uploadRows(c.bg.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(*in.bg)));
    if (in.protect) DI_CUDA_CHECK(c.staging.uploadRows(c.protect.f(), (size_t)w * sizeof(float), h, st, asFill(*in.protect)));
    if (in.mask)    DI_CUDA_CHECK(c.staging.uploadRows(c.mask.f(), (size_t)w * sizeof(float), h, st, asFill(*in.mask)));
    profiler().mark("upload (host->GPU)");

    const dim3 b(16, 16), g = grid2d(w, h, b);
    const float* dKey = in.key ? c.key.f() : nullptr;
    float* pyr0 = c.pyramid.f();

    // 1. local screen
    if (local) {
        kPass1<<<g, b, 0, st>>>(p, c.src.f(), dKey, pyr0);
        pushPull(pyr0, lw, lh, loff, st);
        profiler().mark("local screen");
    }

    // 2. key, screen, core seeds (core written over the pushed screen, pixel by pixel)
    kPass2<<<g, b, 0, st>>>(p, c.src.f(), dKey, in.clean ? c.clean.f() : nullptr, in.protect ? c.protect.f() : nullptr,
                            local ? pyr0 : nullptr, c.screen.f(), c.alpha.f(), c.screenMask.f(), pyr0);
    profiler().mark("key + spill");

    // 3. distance to the screen
    if (p.spatial) {
        int* ny = c.nearestY.as<int>();
        int* nyT = ny + N;
        const unsigned tb = 64;
        const dim3 tt(32, 8);
        kEdtColumns<<<(w + tb - 1) / tb, tb, 0, st>>>(c.screenMask.f(), w, h, ny);
        kTranspose<int><<<dim3((w + 31) / 32, (h + 31) / 32), tt, 0, st>>>(ny, nyT, w, h);
        kEdtRows<<<(h + tb - 1) / tb, tb, 0, st>>>(nyT, w, h, c.edtV.as<int>(), c.edtZ.as<long long>(), c.edtD.as<int>(),
                                                     c.seedT.as<Seed>());
        kTranspose<Seed><<<dim3((h + 31) / 32, (w + 31) / 32), tt, 0, st>>>(c.seedT.as<Seed>(), c.seeds.as<Seed>(), h, w);
        kDistance<<<g, b, 0, st>>>(c.seeds.as<Seed>(), c.dist.f(), w, h);
        profiler().mark("distance to screen");
    }

    // 4. core colour pushed into the edges
    if (core) {
        pushPull(pyr0, lw, lh, loff, st);
        profiler().mark("edge colour");
    }

    // 5. blurred background and key for the light wrap, on an image reduced by rf
    if (bgBlur) {
        kDownsample<<<grid2d(ws, hs, b), b, 0, st>>>(c.bg.f(), w, h, 3, rf, c.bgSmall.f(), ws, hs);
        if (br > 0) blur(c.bgSmall.f(), c.tmpSmall.f(), ws, hs, 3, 3, br, st);
    }
    if (wrap) {
        kDownsample<<<grid2d(ws, hs, b), b, 0, st>>>(c.alpha.f(), w, h, 1, rf, c.alphaSmall.f(), ws, hs);
        if (br > 0) blur(c.alphaSmall.f(), c.tmpSmall.f(), ws, hs, 1, 1, br, st);
    }
    if (bgBlur || wrap) profiler().mark("background blur");

    // 6. final colour
    kFinal<<<g, b, 0, st>>>(p, c.src.f(), c.screen.f(), c.alpha.f(), p.spatial ? c.dist.f() : nullptr,
                            core ? pyr0 : nullptr, bgBlur ? c.bgSmall.f() : nullptr, wrap ? c.alphaSmall.f() : nullptr,
                            ws, hs, rf, in.protect ? c.protect.f() : nullptr, in.mask ? c.mask.f() : nullptr, c.out.f());
    DI_CUDA_CHECK(cudaGetLastError());
    profiler().mark("despill + composite");
    DI_CUDA_CHECK(c.staging.downloadRows(c.out.f(), 4 * (size_t)w * sizeof(float), h, st,
                                         [&](int y0, int y1, const void* i) { out(y0, y1, static_cast<const float*>(i)); }));
    profiler().mark("download (GPU->host)");
    DI_CUDA_CHECK(cudaStreamSynchronize(st));
    profiler().end();
    return true;
}

} // namespace dspit
