// push_IT_OFXCUDA.cu
// CUDA back-end. Nuke does not hand GPU buffers to OFX plug-ins, so the plug-in
// uploads the host images, runs the whole pipeline on the GPU and downloads the
// result. Device buffers are cached between renders and protected by a mutex.
#include "push_IT_OFXCore.h"
#include "PinnedStaging.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace pushit {
namespace {

#define PS_CUDA_CHECK(call)                                                          \
    do {                                                                             \
        cudaError_t e__ = (call);                                                    \
        if (e__ != cudaSuccess) { error = std::string(#call) + " : " + cudaGetErrorString(e__); return false; } \
    } while (0)

// ------------------------------------------------------------- buffer cache
struct DevBuf {
    void*  ptr = nullptr;
    size_t size = 0;
    bool ensure(size_t bytes) {
        if (bytes <= size) return true;
        if (ptr) cudaFree(ptr);
        ptr = nullptr; size = 0;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; return false; }
        size = bytes;
        return true;
    }
    void release() { if (ptr) cudaFree(ptr); ptr = nullptr; size = 0; }
    template <class T> T* as() const { return static_cast<T*>(ptr); }
};

struct DevCache {
    std::mutex mutex;
    DevBuf src, matte, vec, dst, valid, tmp1, dist, edtV, edtZ, edtD, seedT, rgbU, base, seedA, seedB, fill, tmp4, pyramid;
    PinnedStaging staging;
    cudaStream_t stream = nullptr;
    void release() {
        DevBuf* all[] = { &src, &matte, &vec, &dst, &valid, &tmp1, &dist, &edtV, &edtZ, &edtD, &seedT, &rgbU, &base, &seedA, &seedB, &fill, &tmp4, &pyramid };
        for (DevBuf* b : all) b->release();
        staging.release();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};

DevCache& cache() { static DevCache* c = new DevCache(); return *c; }   // intentionally leaked (DLL unload order)

// ------------------------------------------------------------ stage profiler
// Optional per-stage GPU timings (CUDA events), used by the benchmark tool only.
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
    void end()   // after the stream is synchronised
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
__global__ void kValid(Params p, const float* src, const float* matte, float* valid)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    valid[i] = computeValid(p, matte ? matte[i] : src[4 * i + 3]);
}

// Separable min filter (erosion), samples outside the image are ignored.
__global__ void kMin1D(const float* in, float* out, int w, int h, int r, int horizontal)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float m = in[(size_t)y * w + x];
    if (horizontal) {
        const int a = max(0, x - r), b = min(w - 1, x + r);
        for (int k = a; k <= b; ++k) m = fminf(m, in[(size_t)y * w + k]);
    } else {
        const int a = max(0, y - r), b = min(h - 1, y + r);
        for (int k = a; k <= b; ++k) m = fminf(m, in[(size_t)k * w + x]);
    }
    out[(size_t)y * w + x] = m;
}

// Sliding box along columns, split into segments so that many threads work on each
// column (one thread per whole line left most of the GPU idle). Each segment starts
// with a full window sum, then slides like slideBox(); neighbouring threads handle
// neighbouring columns, so the reads are coalesced. Rows are done on a transposed copy.
// Channels >= nch are copied through unchanged.
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
        const float* q = a + (size_t)psClampi(i, 0, h - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += q[c];
    }
    for (int y = y0; y < y1; ++y) {
        const float* q = a + (size_t)y * stride;
        float* d = o + (size_t)y * stride;
        for (int c = 0; c < nch; ++c) d[c] = (float)(sum[c] * norm);
        for (int c = nch; c < pstride; ++c) d[c] = q[c];
        const float* qa = a + (size_t)psClampi(y + r + 1, 0, h - 1) * stride;
        const float* qs = a + (size_t)psClampi(y - r, 0, h - 1) * stride;
        for (int c = 0; c < nch; ++c) sum[c] += (double)qa[c] - (double)qs[c];
    }
}

// Tiled transpose of a w x h image of T (float or float4) into an h x w image.
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
        const int x = by + threadIdx.x, y = bx + j;   // out is h wide
        if (x < h && y < w) out[(size_t)y * h + x] = tile[threadIdx.x][j];
    }
}

// Exact distance transform: pass 1 (one thread per column), pass 2 (one thread per row).
__global__ void kEdtColumns(Params p, const float* valid, int* nearestY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= p.width) return;
    edtColumn(valid, p.width, p.height, x, effectiveSeedThreshold(p), nearestY);
}

// Pass 2 on a transposed nearestY (column q of the image = row q of nyT): one thread
// per image row, with all per-row arrays interleaved (element i of row y at i * h + y)
// so that neighbouring threads read neighbouring addresses. Seeds come out transposed.
__global__ void kEdtRows(const int* nyT, int w, int h, int* v, long long* zn, int* zd, Seed* seedsT)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;
    edtRow(nyT + y, w, y, (size_t)h, v + y, zn + y, zd + y, seedsT + y);
}

__global__ void kBase(Params p, const float* src, const float* valid, float* rgbU, float* base)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    unpremultSource(p, src + 4 * i, rgbU + 4 * i);
    buildBase(rgbU + 4 * i, valid[i], base + 4 * i);
}


__global__ void kDistance(const Seed* seeds, float* dist, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x;
    dist[i] = seedDistance(seeds[i], x, y);
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

__global__ void kFill(Params p, const float* rgbU, const float* base, const float* valid,
                      const Seed* seeds, const float* distField, const float* pushed, const float* vec, float* fill)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = p.width, h = p.height;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x;
    float* o = fill + 4 * i;
    Seed s; s.x = -1; s.y = -1;
    if (seeds) s = seeds[i];
    const bool hasSeed = s.x >= 0;
    const float dist = hasSeed ? distField[i] : 0.f;
    bool hasFill = false;
    if (p.spreadType == kTypeStretch) {
        hasFill = hasSeed;
        if (hasSeed) {
            const float* c = rgbU + 4 * ((size_t)s.y * w + s.x);
            o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
        } else { o[0] = o[1] = o[2] = 0.f; }
    } else {
        hasFill = resolveWeighted(pushed + 4 * i, o);
        if (p.spreadType == kTypeVectorWarp && valid[i] < 1.f) {
            float dx, dy, C[3];
            spreadVector(p, vec ? vec + 4 * i : nullptr, &dx, &dy);
            const float A = marchPixel(p, base, distField, w, h, x, y, dx, dy, C);
            if (A > 0.f) {
                const float k = 1.f - A;
                o[0] = C[0] + k * o[0]; o[1] = C[1] + k * o[1]; o[2] = C[2] + k * o[2];
                hasFill = true;
            }
        }
    }
    o[3] = spreadMatte(p, hasFill, hasSeed, dist);
}

__global__ void kComposite(Params p, const float* rgbU, const float* valid, const float* fill, float* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    compositePixel(p, rgbU + 4 * i, valid[i], fill + 4 * i, fill[4 * i + 3], dst + 4 * i);
}

void transposeImage(const float* in, float* out, int w, int h, int pstride, cudaStream_t st)
{
    const dim3 tb(32, 8), tg((w + 31) / 32, (h + 31) / 32);
    if (pstride == 4) kTranspose<float4><<<tg, tb, 0, st>>>((const float4*)in, (float4*)out, w, h);
    else              kTranspose<float><<<tg, tb, 0, st>>>(in, out, w, h);
}

// 3 column passes of a w x h image: in -> other -> in -> other... ends in `b` (passes: a->b, b->a, a->b).
void slideColumns3(float* a, float* b, int w, int h, int pstride, int nch, int r, cudaStream_t st)
{
    const int seg = (2 * r + 1) > 48 ? (2 * r + 1) : 48;   // init cost of a segment <= its length
    const unsigned tb = 128;
    const dim3 g((w + tb - 1) / tb, (h + seg - 1) / seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, pstride, nch, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(b, a, w, h, pstride, nch, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, pstride, nch, r, seg);
}

// 3 box passes per direction (rows first, like the CPU) in place on `img`, `tmp` as scratch.
void gaussianish(float* img, float* tmp, int w, int h, int pstride, int nch, float size, cudaStream_t st)
{
    const int r = boxRadiusForBlur(size);
    if (r <= 0) return;
    transposeImage(img, tmp, w, h, pstride, st);           // tmp: h x w
    slideColumns3(tmp, img, h, w, pstride, nch, r, st);   // rows of the image -> result in img (h x w)
    transposeImage(img, tmp, h, w, pstride, st);           // tmp: w x h
    slideColumns3(tmp, img, w, h, pstride, nch, r, st);   // columns -> result in img
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

bool processCUDA(const Params& p, const float* src, const float* matte,
                 const float* vec, float* dst, std::string& error)
{
    const size_t w = (size_t)p.width;
    const RowReader rs = [&](int y0, int y1, float* o) { std::memcpy(o, src + 4 * w * y0, 4 * w * (y1 - y0) * sizeof(float)); };
    const RowReader rm = [&](int y0, int y1, float* o) { std::memcpy(o, matte + w * y0, w * (y1 - y0) * sizeof(float)); };
    const RowReader rv = [&](int y0, int y1, float* o) { std::memcpy(o, vec + 4 * w * y0, 4 * w * (y1 - y0) * sizeof(float)); };
    const RowWriter wd = [&](int y0, int y1, const float* i) { std::memcpy(dst + 4 * w * y0, i, 4 * w * (y1 - y0) * sizeof(float)); };
    return processCUDARows(p, rs, matte ? &rm : nullptr, vec ? &rv : nullptr, wd, error);
}

bool processCUDARows(const Params& p, const RowReader& src, const RowReader* matte,
                     const RowReader* vec, const RowWriter& dst, std::string& error)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0) return true;
    const size_t N = (size_t)w * h;
    const size_t f4 = 4 * N * sizeof(float), f1 = N * sizeof(float);

    DevCache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.stream) PS_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaStream_t st = c.stream;

    // pyramid size (levels 1..n, level 0 lives in `base` copy -> `pushed`)
    std::vector<int> lw(1, w), lh(1, h);
    std::vector<size_t> loff(1, 0);
    size_t pyr = f4;   // level 0
    while (lw.back() > 1 || lh.back() > 1) {
        lw.push_back((lw.back() + 1) / 2);
        lh.push_back((lh.back() + 1) / 2);
        loff.push_back(pyr / sizeof(float));
        pyr += (size_t)4 * lw.back() * lh.back() * sizeof(float);
    }

    bool ok = c.src.ensure(f4) && c.dst.ensure(f4) && c.valid.ensure(f1) && c.tmp1.ensure(f1) && c.dist.ensure(f1) &&
              c.rgbU.ensure(f4) && c.base.ensure(f4) && c.fill.ensure(f4) && c.tmp4.ensure(f4) &&
              c.seedA.ensure(N * sizeof(Seed)) && c.seedB.ensure(N * sizeof(Seed)) &&
              c.edtV.ensure(N * sizeof(int)) && c.edtZ.ensure(N * sizeof(long long)) &&
              c.edtD.ensure(N * sizeof(int)) && c.seedT.ensure(N * sizeof(Seed)) &&
              (!matte || c.matte.ensure(f1)) && (!vec || c.vec.ensure(f4)) &&
              (p.spreadType == kTypeStretch || c.pyramid.ensure(pyr));
    if (!ok) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }

    float* dSrc = c.src.as<float>();
    float* dMatte = matte ? c.matte.as<float>() : nullptr;
    float* dVec = vec ? c.vec.as<float>() : nullptr;
    profiler().begin(st);
    auto asFill = [](const RowReader& r) {
        return [&r](int y0, int y1, void* d) { r(y0, y1, static_cast<float*>(d)); };
    };
    PS_CUDA_CHECK(c.staging.uploadRows(dSrc, 4 * (size_t)w * sizeof(float), h, st, asFill(src)));
    if (matte) PS_CUDA_CHECK(c.staging.uploadRows(dMatte, (size_t)w * sizeof(float), h, st, asFill(*matte)));
    if (vec)   PS_CUDA_CHECK(c.staging.uploadRows(dVec, 4 * (size_t)w * sizeof(float), h, st, asFill(*vec)));
    profiler().mark("upload (host->GPU)");

    const dim3 b(16, 16), g = grid2d(w, h, b);
    float* dValid = c.valid.as<float>();
    float* dTmp1 = c.tmp1.as<float>();

    // 1. valid area
    kValid<<<g, b, 0, st>>>(p, dSrc, dMatte, dValid);
    const int er = (int)lroundf(p.shrink);
    if (er > 0) {
        kMin1D<<<g, b, 0, st>>>(dValid, dTmp1, w, h, er, 1);
        kMin1D<<<g, b, 0, st>>>(dTmp1, dValid, w, h, er, 0);
    }
    if (p.edgeSoftness > 0.f) gaussianish(dValid, dTmp1, w, h, 1, 1, p.edgeSoftness, st);

    // 2. base
    float* dRgbU = c.rgbU.as<float>();
    float* dBase = c.base.as<float>();
    kBase<<<g, b, 0, st>>>(p, dSrc, dValid, dRgbU, dBase);
    profiler().mark("matte/shrink/soft/base");

    // 3. nearest seed: exact distance transform
    const bool needSeeds = p.spreadType != kTypeInterpolate || p.maxDistance > 0.f;
    Seed* sIn = c.seedA.as<Seed>();
    if (needSeeds) {
        int* nearestY = c.seedB.as<int>();         // seedB holds 2 int images: nearestY
        int* nearestYT = nearestY + N;               // and its transpose
        const unsigned tb = 64;
        const dim3 tt(32, 8);
        kEdtColumns<<<(w + tb - 1) / tb, tb, 0, st>>>(p, dValid, nearestY);
        kTranspose<int><<<dim3((w + 31) / 32, (h + 31) / 32), tt, 0, st>>>(nearestY, nearestYT, w, h);
        kEdtRows<<<(h + tb - 1) / tb, tb, 0, st>>>(nearestYT, w, h, c.edtV.as<int>(), c.edtZ.as<long long>(),
                                                     c.edtD.as<int>(), c.seedT.as<Seed>());
        kTranspose<Seed><<<dim3((h + 31) / 32, (w + 31) / 32), tt, 0, st>>>(c.seedT.as<Seed>(), sIn, h, w);
        kDistance<<<g, b, 0, st>>>(sIn, c.dist.as<float>(), w, h);
    }

    profiler().mark("distance transform");

    // 4. push-pull
    float* dPushed = nullptr;
    if (p.spreadType != kTypeStretch) {
        float* pyrBase = c.pyramid.as<float>();
        PS_CUDA_CHECK(cudaMemcpyAsync(pyrBase, dBase, f4, cudaMemcpyDeviceToDevice, st));
        for (size_t l = 1; l < lw.size(); ++l) {
            const dim3 gl = grid2d(lw[l], lh[l], b);
            kPull<<<gl, b, 0, st>>>(pyrBase + loff[l - 1], lw[l - 1], lh[l - 1], pyrBase + loff[l], lw[l], lh[l]);
        }
        for (int l = (int)lw.size() - 2; l >= 0; --l) {
            const dim3 gl = grid2d(lw[l], lh[l], b);
            kPush<<<gl, b, 0, st>>>(pyrBase + loff[l], lw[l], lh[l], pyrBase + loff[l + 1], lw[l + 1], lh[l + 1]);
        }
        dPushed = pyrBase;
    }

    profiler().mark("push-pull pyramid");

    // 5. fill + spread matte
    float* dFill = c.fill.as<float>();
    kFill<<<g, b, 0, st>>>(p, dRgbU, dBase, dValid, needSeeds ? sIn : nullptr,
                           needSeeds ? c.dist.as<float>() : nullptr, dPushed, dVec, dFill);

    profiler().mark("fill (+vector march)");

    // 6. spread blur
    if (p.blur > 0.f) gaussianish(dFill, c.tmp4.as<float>(), w, h, 4, 3, p.blur, st);

    profiler().mark("spread blur");

    // 7. composite + download
    float* dDst = c.dst.as<float>();
    kComposite<<<g, b, 0, st>>>(p, dRgbU, dValid, dFill, dDst);
    PS_CUDA_CHECK(cudaGetLastError());
    profiler().mark("composite");
    PS_CUDA_CHECK(c.staging.downloadRows(dDst, 4 * (size_t)w * sizeof(float), h, st,
                                         [&](int y0, int y1, const void* i) { dst(y0, y1, static_cast<const float*>(i)); }));
    profiler().mark("download (GPU->host)");
    PS_CUDA_CHECK(cudaStreamSynchronize(st));
    profiler().end();
    return true;
}

} // namespace pushit
