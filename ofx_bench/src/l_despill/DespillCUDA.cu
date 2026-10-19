// DespillCUDA.cu
// CUDA back-end of L_Despill_OFX. Nuke hands host images to OFX plug-ins: the rows are
// written straight into page-locked transfer memory, the whole effect runs on the GPU and
// the result rows are read straight out of it. Device buffers are cached (mutex protected).
// Same arithmetic as the CPU back-end (built without fast-math and without FMA contraction).
#include "DespillCore.h"
#include "PinnedStaging.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ldespill {
namespace {

#define LD_CUDA_CHECK(call)                                                          \
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
    float* f() const { return static_cast<float*>(ptr); }
};

struct DevCache {
    std::mutex mutex;
    DevBuf src, bg, mask, matte, tmp, out;
    PinnedStaging staging;
    cudaStream_t stream = nullptr;
    void release() {
        DevBuf* all[] = { &src, &bg, &mask, &matte, &tmp, &out };
        for (DevBuf* b : all) b->release();
        staging.release();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};
DevCache& cache() { static DevCache* c = new DevCache(); return *c; }   // leaked on purpose (DLL unload order)

// ------------------------------------------------------------ stage profiler
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
__global__ void kMatte(Params p, const float* src, float* matte)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.bufW || y >= p.bufH) return;
    const size_t i = (size_t)y * p.bufW + x;
    matte[i] = matteRawPixel(p, src + 4 * i);
}

// Median of the (2r+1)^2 window (r <= kMaxMedian), edges repeated. Quickselect on a local
// copy: the k-th value is unique, so the result is exactly the CPU's nth_element.
constexpr int kMaxMedian = 10;
__global__ void kMedian(const float* in, float* out, int w, int h, int r)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float v[(2 * kMaxMedian + 1) * (2 * kMaxMedian + 1)];
    int n = 0;
    for (int dy = -r; dy <= r; ++dy) {
        const float* row = in + (size_t)ldClampi(y + dy, 0, h - 1) * w;
        for (int dx = -r; dx <= r; ++dx) v[n++] = row[ldClampi(x + dx, 0, w - 1)];
    }
    const int k = n / 2;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const float pivot = v[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] < pivot) ++i;
            while (v[j] > pivot) --j;
            if (i <= j) { const float t = v[i]; v[i] = v[j]; v[j] = t; ++i; --j; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    out[(size_t)y * w + x] = v[k];
}

// Min / max over 2r+1 pixels along a row (dx = 1) or a column (dx = 0), edges repeated.
__global__ void kMinMax(const float* in, float* out, int w, int h, int r, int dilate, int alongRows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float m = alongRows ? in[(size_t)y * w + ldClampi(x - r, 0, w - 1)] : in[(size_t)ldClampi(y - r, 0, h - 1) * w + x];
    for (int d = -r + 1; d <= r; ++d) {
        const float v = alongRows ? in[(size_t)y * w + ldClampi(x + d, 0, w - 1)] : in[(size_t)ldClampi(y + d, 0, h - 1) * w + x];
        m = dilate ? (v > m ? v : m) : (v < m ? v : m);
    }
    out[(size_t)y * w + x] = m;
}

// Sliding box along columns, split into segments so that many threads work on each column.
// Each segment starts with a full window sum, then slides like slideBox(). Neighbouring
// threads handle neighbouring columns (coalesced reads); rows are done on a transposed copy.
__global__ void kSlideSegments(const float* in, float* out, int w, int h, int r, int seg)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, sg = blockIdx.y;
    if (x >= w) return;
    const int y0 = sg * seg, y1 = y0 + seg < h ? y0 + seg : h;
    if (y0 >= h) return;
    const double norm = 1.0 / (double)(2 * r + 1);
    double sum = 0.0;
    for (int i = y0 - r; i <= y0 + r; ++i) sum += in[(size_t)ldClampi(i, 0, h - 1) * w + x];
    for (int y = y0; y < y1; ++y) {
        out[(size_t)y * w + x] = (float)(sum * norm);
        sum += (double)in[(size_t)ldClampi(y + r + 1, 0, h - 1) * w + x] - (double)in[(size_t)ldClampi(y - r, 0, h - 1) * w + x];
    }
}

// Tiled transpose of a w x h image into an h x w image.
__global__ void kTranspose(const float* in, float* out, int w, int h)
{
    __shared__ float tile[32][33];
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

__global__ void kFinal(Params p, const float* src, const float* bg, const float* mask, const float* matte, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.winW || y >= p.winH) return;
    const size_t bi = (size_t)(y + p.winY) * p.bufW + (x + p.winX);
    const size_t wi = (size_t)y * p.winW + x;
    finalPixel(p, src + 4 * bi, bg ? bg + 3 * wi : nullptr, matte[bi], mask ? mask[wi] : 1.f, out + 4 * wi);
}

void transpose(const float* in, float* out, int w, int h, cudaStream_t st)
{
    kTranspose<<<dim3((w + 31) / 32, (h + 31) / 32), dim3(32, 8), 0, st>>>(in, out, w, h);
}

// 3 column passes a -> b -> a -> b.
void slideColumns3(float* a, float* b, int w, int h, int r, cudaStream_t st)
{
    const int seg = (2 * r + 1) > 48 ? (2 * r + 1) : 48;
    const unsigned tb = 128;
    const dim3 g((w + tb - 1) / tb, (h + seg - 1) / seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(b, a, w, h, r, seg);
    kSlideSegments<<<g, tb, 0, st>>>(a, b, w, h, r, seg);
}

// 3 box passes along the rows then along the columns, in place on img (tmp: scratch).
void blur(float* img, float* tmp, int w, int h, int r, cudaStream_t st)
{
    transpose(img, tmp, w, h, st);         // tmp: h x w
    slideColumns3(tmp, img, h, w, r, st);  // rows -> img (h x w)
    transpose(img, tmp, h, w, st);         // tmp: w x h
    slideColumns3(tmp, img, w, h, r, st);  // columns -> img
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

bool processCUDA(const Params& p, const float* src, const float* bg, const float* mask, float* out, std::string& error)
{
    const size_t bw = (size_t)p.bufW, ww = (size_t)p.winW;
    const RowReader rs = [&](int y0, int y1, float* o) { std::memcpy(o, src + 4 * bw * y0, 4 * bw * (y1 - y0) * sizeof(float)); };
    const RowReader rb = [&](int y0, int y1, float* o) { std::memcpy(o, bg + 3 * ww * y0, 3 * ww * (y1 - y0) * sizeof(float)); };
    const RowReader rm = [&](int y0, int y1, float* o) { std::memcpy(o, mask + ww * y0, ww * (y1 - y0) * sizeof(float)); };
    const RowWriter wo = [&](int y0, int y1, const float* i) { std::memcpy(out + 4 * ww * y0, i, 4 * ww * (y1 - y0) * sizeof(float)); };
    return processCUDARows(p, rs, bg ? &rb : nullptr, mask ? &rm : nullptr, wo, error);
}

bool processCUDARows(const Params& p, const RowReader& src, const RowReader* bg, const RowReader* mask,
                     const RowWriter& out, std::string& error)
{
    const int w = p.bufW, h = p.bufH;
    if (w <= 0 || h <= 0 || p.winW <= 0 || p.winH <= 0) return true;
    if (p.medianRadius > kMaxMedian) { error = "median radius too large for the GPU"; return false; }
    const size_t N = (size_t)w * h, NW = (size_t)p.winW * p.winH;

    DevCache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.stream) LD_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaStream_t st = c.stream;

    const bool ok = c.src.ensure(4 * N * sizeof(float)) && c.matte.ensure(N * sizeof(float)) &&
                    c.tmp.ensure(N * sizeof(float)) && c.out.ensure(4 * NW * sizeof(float)) &&
                    (!bg || c.bg.ensure(3 * NW * sizeof(float))) && (!mask || c.mask.ensure(NW * sizeof(float)));
    if (!ok) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }

    auto asFill = [](const RowReader& r) {
        return [&r](int y0, int y1, void* d) { r(y0, y1, static_cast<float*>(d)); };
    };
    profiler().begin(st);
    LD_CUDA_CHECK(c.staging.uploadRows(c.src.f(), 4 * (size_t)w * sizeof(float), h, st, asFill(src)));
    if (bg)   LD_CUDA_CHECK(c.staging.uploadRows(c.bg.f(), 3 * (size_t)p.winW * sizeof(float), p.winH, st, asFill(*bg)));
    if (mask) LD_CUDA_CHECK(c.staging.uploadRows(c.mask.f(), (size_t)p.winW * sizeof(float), p.winH, st, asFill(*mask)));
    profiler().mark("upload (host->GPU)");

    const dim3 b(16, 16), g = grid2d(w, h, b);
    float* m = c.matte.f();
    float* t = c.tmp.f();
    kMatte<<<g, b, 0, st>>>(p, c.src.f(), m);
    profiler().mark("spill matte");

    if (p.medianRadius > 0) {
        kMedian<<<g, b, 0, st>>>(m, t, w, h, p.medianRadius);
        std::swap(m, t);
        profiler().mark("median");
    }
    if (p.cleanRadius != 0) {
        const int r = p.cleanRadius < 0 ? -p.cleanRadius : p.cleanRadius;
        const int first = p.cleanRadius > 0 ? 1 : 0;       // close: dilate then erode / open: erode then dilate
        for (int pass = 0; pass < 2; ++pass) {
            const int dil = pass == 0 ? first : 1 - first;
            kMinMax<<<g, b, 0, st>>>(m, t, w, h, r, dil, 1);
            kMinMax<<<g, b, 0, st>>>(t, m, w, h, r, dil, 0);
        }
        profiler().mark("clean");
    }
    const int br = boxRadiusForBlur(p.matteBlur);
    if (br > 0) {
        blur(m, t, w, h, br, st);
        profiler().mark("matte blur");
    }

    kFinal<<<grid2d(p.winW, p.winH, b), b, 0, st>>>(p, c.src.f(), bg ? c.bg.f() : nullptr, mask ? c.mask.f() : nullptr,
                                                   m, c.out.f());
    LD_CUDA_CHECK(cudaGetLastError());
    profiler().mark("despill + grades");
    LD_CUDA_CHECK(c.staging.downloadRows(c.out.f(), 4 * (size_t)p.winW * sizeof(float), p.winH, st,
                                         [&](int y0, int y1, const void* i) { out(y0, y1, static_cast<const float*>(i)); }));
    profiler().mark("download (GPU->host)");
    LD_CUDA_CHECK(cudaStreamSynchronize(st));
    profiler().end();
    return true;
}

} // namespace ldespill
