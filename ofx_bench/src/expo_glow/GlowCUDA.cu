// GlowCUDA.cu
// CUDA back-end of expo_GLOW_OFX. Nuke hands host images to OFX plug-ins, so the
// image is uploaded, the whole glow is computed on the GPU and the result is
// downloaded. Device buffers are cached between renders (mutex protected).
//
// Every blur pass runs one thread per line with the same sliding-window code as
// the CPU (boxLine). Horizontal passes are done on a transposed copy so that
// neighbouring threads always read neighbouring addresses.
#include "GlowCore.h"
#include "PinnedStaging.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

namespace expoglow {
namespace {

#define EG_CUDA_CHECK(call)                                                          \
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
    DevBuf src, area, prep, glow, reduced, padded, bufA, bufB, blurred;
    DevBuf rowsReduced;                 // downsample: rows reduced, columns not yet
    DevBuf packed;                      // RGB of the output window, for the download
    PinnedStaging staging;
    cudaStream_t stream = nullptr;
    void release() {
        DevBuf* all[] = { &src, &area, &prep, &glow, &reduced, &padded, &bufA, &bufB, &blurred, &rowsReduced, &packed };
        for (DevBuf* b : all) b->release();
        staging.release();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
};

DevCache& cache() { static DevCache* c = new DevCache(); return *c; }   // leaked on purpose (DLL unload order)

// ------------------------------------------------------------------- kernels
__global__ void kPrep(PrepParams p, const float* srcRGB, const float* area, float* prep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;
    const size_t i = (size_t)y * p.width + x;
    prepPixel(p, srcRGB + 3 * i, area ? area[i] : 1.f, prep + 4 * i);   // reads r, g, b only
}

// One thread per column of a w x h RGBA image: vertical fractional box.
__global__ void kBoxColumns(const float* in, float* out, int w, int h, float r)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w) return;
    boxLine(in + 4 * (size_t)x, out + 4 * (size_t)x, h, 4 * (size_t)w, r);
}

// RGBA transpose through shared memory: in is w x h, out is h x w.
__global__ void kTranspose(const float4* in, float4* out, int w, int h)
{
    __shared__ float4 tile[16][17];
    int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
    if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[(size_t)y * w + x];
    __syncthreads();
    x = blockIdx.y * 16 + threadIdx.x;   // column in out (= row in in)
    y = blockIdx.x * 16 + threadIdx.y;   // row in out (= column in in)
    if (x < h && y < w) out[(size_t)y * h + x] = tile[threadIdx.x][threadIdx.y];
}

// Area-average downsample by f, in two separable passes so that every thread only sums
// f pixels (one thread per f x f block was very slow for large factors: few threads,
// f*f serial reads each). Same result as downsamplePixel() up to float rounding.
__global__ void kDownsampleRows(const float* in, int w, int h, int f, float* out, int ws)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= ws || y >= h) return;
    const int x0 = X * f, x1 = x0 + f < w ? x0 + f : w;
    const float4* q = reinterpret_cast<const float4*>(in) + (size_t)y * w;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int x = x0; x < x1; ++x) { const float4 v = q[x]; s0 += v.x; s1 += v.y; s2 += v.z; }
    const double k = 1.0 / (double)(x1 - x0);
    reinterpret_cast<float4*>(out)[(size_t)y * ws + X] = make_float4((float)(s0 * k), (float)(s1 * k), (float)(s2 * k), 0.f);
}

__global__ void kDownsampleCols(const float* in, int ws, int h, int f, float* out, int hs)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= ws || Y >= hs) return;
    const int y0 = Y * f, y1 = y0 + f < h ? y0 + f : h;
    const float4* q = reinterpret_cast<const float4*>(in);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int y = y0; y < y1; ++y) { const float4 v = q[(size_t)y * ws + X]; s0 += v.x; s1 += v.y; s2 += v.z; }
    const double k = 1.0 / (double)(y1 - y0);
    reinterpret_cast<float4*>(out)[(size_t)Y * ws + X] = make_float4((float)(s0 * k), (float)(s1 * k), (float)(s2 * k), 0.f);
}

// RGB of the window (ox, oy, ow, oh) of the RGBA glow, packed for the download.
__global__ void kPackRGB(const float* glow, int bw, int ox, int oy, int ow, int oh, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    const float4 g = reinterpret_cast<const float4*>(glow)[(size_t)(y + oy) * bw + (x + ox)];
    float* o = out + 3 * ((size_t)y * ow + x);
    o[0] = g.x; o[1] = g.y; o[2] = g.z;
}

__global__ void kUpsampleAccumulate(const float* small, int ws, int hs, int f, float* glow, int w, int h,
                                    float w0, float w1, float w2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float w3[3] = { w0, w1, w2 };
    upsampleAccumulate(small, ws, hs, f, x, y, glow + 4 * ((size_t)y * w + x), w3);
}

__global__ void kPad(const float* in, int w, int h, int pad, float* out, int pw, int ph)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) return;
    padPixel(in, w, h, pad, x, y, out + 4 * ((size_t)y * pw + x));
}

__global__ void kCrop(const float* in, int pw, int pad, float* out, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* q = in + 4 * ((size_t)(y + pad) * pw + x + pad);
    float* o = out + 4 * ((size_t)y * w + x);
    o[0] = q[0]; o[1] = q[1]; o[2] = q[2]; o[3] = 0.f;
}

// Merged kernel, one pass per direction (edge pixels repeated = exact extension).
struct KernelArg { int taps; float k[kMaxTaps]; };

__global__ void kConvRows(const float* in, float* out, int w, int h, KernelArg ka)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    convPixel(in + 4 * (size_t)y * w, w, 4, x, ka.k, ka.taps, out + 4 * ((size_t)y * w + x));
}

__global__ void kConvCols(const float* in, float* out, int w, int h, KernelArg ka)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    convPixel(in + 4 * (size_t)x, h, 4 * (size_t)w, y, ka.k, ka.taps, out + 4 * ((size_t)y * w + x));
}

__global__ void kAccumulate(float* glow, const float* blur, float w0, float w1, float w2, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w3[3] = { w0, w1, w2 };
    accumulatePixel(glow + 4 * i, blur + 4 * i, w3);
}

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
inline unsigned blocks1d(size_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }

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

bool computeGlowCUDA(const PrepParams& p, const std::vector<StepPlan>& plan,
                     const float* src, const float* area, float* glow, std::string& error)
{
    const size_t w = (size_t)p.width;
    const RowReader rs = [&](int y0, int y1, float* o) {
        const float* s = src + 4 * w * y0;
        for (size_t i = 0, n = w * (y1 - y0); i < n; ++i, s += 4, o += 3) { o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; }
    };
    const RowReader ra = [&](int y0, int y1, float* o) { std::memcpy(o, area + w * y0, w * (y1 - y0) * sizeof(float)); };
    const RowWriter wg = [&](int y0, int y1, const float* i) {
        float* o = glow + 4 * w * y0;
        for (size_t k = 0, n = w * (y1 - y0); k < n; ++k, i += 3, o += 4) { o[0] = i[0]; o[1] = i[1]; o[2] = i[2]; o[3] = 0.f; }
    };
    return computeGlowCUDARows(p, plan, rs, area ? &ra : nullptr, 0, 0, p.width, p.height, wg, error);
}

bool computeGlowCUDARows(const PrepParams& p, const std::vector<StepPlan>& plan,
                         const RowReader& src, const RowReader* area,
                         int outX, int outY, int outW, int outH, const RowWriter& glowOut, std::string& error)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0) return true;
    const size_t N = (size_t)w * h;
    const size_t f4 = 4 * N * sizeof(float), f1 = N * sizeof(float);

    DevCache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.stream) EG_CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaStream_t st = c.stream;

    const size_t f3 = 3 * N * sizeof(float);
    const bool ok = c.src.ensure(f3) && c.prep.ensure(f4) && c.glow.ensure(f4) && (!area || c.area.ensure(f1)) &&
                    c.packed.ensure(3 * (size_t)std::max(outW, 0) * std::max(outH, 0) * sizeof(float) + 16);
    if (!ok) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }

    float* dArea = area ? c.area.f() : nullptr;
    profiler().begin(st);
    auto asFill = [](const RowReader& r) {
        return [&r](int y0, int y1, void* d) { r(y0, y1, static_cast<float*>(d)); };
    };
    EG_CUDA_CHECK(c.staging.uploadRows(c.src.f(), 3 * (size_t)w * sizeof(float), h, st, asFill(src)));
    if (area) EG_CUDA_CHECK(c.staging.uploadRows(dArea, (size_t)w * sizeof(float), h, st, asFill(*area)));
    EG_CUDA_CHECK(cudaMemsetAsync(c.glow.f(), 0, f4, st));
    profiler().mark("upload (host->GPU)");

    const dim3 b2(16, 16);
    kPrep<<<grid2d(w, h, b2), b2, 0, st>>>(p, c.src.f(), dArea, c.prep.f());
    profiler().mark("prep (response)");

    const unsigned tb = 128;
    // Blur of a bw x bh image into c.blurred (same arithmetic as the CPU version).
    auto blur = [&](const float* in, int bw, int bh, const StepPlan& step) -> bool {
        const size_t n4 = 4 * (size_t)bw * bh * sizeof(float);
        if (!c.blurred.ensure(n4)) return false;
        if (step.taps > 0) {
            if (!c.bufA.ensure(n4)) return false;
            KernelArg ka;
            ka.taps = step.taps;
            for (int t = 0; t < kMaxTaps; ++t) ka.k[t] = t < step.taps ? step.kernel[t] : 0.f;
            kConvRows<<<grid2d(bw, bh, b2), b2, 0, st>>>(in, c.bufA.f(), bw, bh, ka);
            kConvCols<<<grid2d(bw, bh, b2), b2, 0, st>>>(c.bufA.f(), c.blurred.f(), bw, bh, ka);
            return true;
        }
        const int pad = planPadding(step);
        const int pw = bw + 2 * pad, ph = bh + 2 * pad;
        const size_t pn4 = 4 * (size_t)pw * ph * sizeof(float);
        if (!c.padded.ensure(pn4) || !c.bufA.ensure(pn4) || !c.bufB.ensure(pn4)) return false;
        kPad<<<grid2d(pw, ph, b2), b2, 0, st>>>(in, bw, bh, pad, c.padded.f(), pw, ph);
        // horizontal passes = column passes on the transposed image (ph wide, pw tall)
        kTranspose<<<grid2d(pw, ph, b2), b2, 0, st>>>((const float4*)c.padded.f(), (float4*)c.bufA.f(), pw, ph);
        float* outs[2] = { c.bufB.f(), c.bufA.f() };
        const float* t = c.bufA.f();
        int k = 0;
        for (int pass = 0; pass < step.passes; ++pass) {
            float* o = outs[k++ & 1];
            kBoxColumns<<<blocks1d(ph, tb), tb, 0, st>>>(t, o, ph, pw, step.radius[pass]);
            t = o;
        }
        kTranspose<<<grid2d(ph, pw, b2), b2, 0, st>>>((const float4*)t, (float4*)c.padded.f(), ph, pw);
        const float* cur = c.padded.f();
        float* vouts[2] = { c.bufA.f(), c.bufB.f() };
        for (int pass = 0; pass < step.passes; ++pass) {
            float* o = vouts[pass & 1];
            kBoxColumns<<<blocks1d(pw, tb), tb, 0, st>>>(cur, o, pw, ph, step.radius[pass]);
            cur = o;
        }
        kCrop<<<grid2d(bw, bh, b2), b2, 0, st>>>(cur, pw, pad, c.blurred.f(), bw, bh);
        return true;
    };

    int stepIndex = 0;
    for (const StepPlan& step : plan) {
        ++stepIndex;
        if (step.factor > 1) {
            const int fct = step.factor;
            const int ws = (w + fct - 1) / fct, hs = (h + fct - 1) / fct;
            if (!c.reduced.ensure(4 * (size_t)ws * hs * sizeof(float))) { error = "Not enough GPU memory"; return false; }
            if (!c.rowsReduced.ensure(4 * (size_t)ws * h * sizeof(float))) { error = "Not enough GPU memory"; return false; }
            const dim3 bd(8, 32);
            kDownsampleRows<<<grid2d(ws, h, bd), bd, 0, st>>>(c.prep.f(), w, h, fct, c.rowsReduced.f(), ws);
            kDownsampleCols<<<grid2d(ws, hs, b2), b2, 0, st>>>(c.rowsReduced.f(), ws, h, fct, c.reduced.f(), hs);
            if (!blur(c.reduced.f(), ws, hs, step)) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }
            kUpsampleAccumulate<<<grid2d(w, h, b2), b2, 0, st>>>(c.blurred.f(), ws, hs, fct, c.glow.f(), w, h,
                                                               step.weight[0], step.weight[1], step.weight[2]);
        } else {
            if (!blur(c.prep.f(), w, h, step)) { cudaGetLastError(); error = "Not enough GPU memory"; return false; }
            kAccumulate<<<blocks1d(N, 256), 256, 0, st>>>(c.glow.f(), c.blurred.f(), step.weight[0], step.weight[1], step.weight[2], N);
        }
        profiler().mark("step " + std::to_string(stepIndex) + (step.factor > 1 ? " (reduced /" + std::to_string(step.factor) + ")" : " (full res)"));
    }

    EG_CUDA_CHECK(cudaGetLastError());
    if (outW > 0 && outH > 0) {
        kPackRGB<<<grid2d(outW, outH, b2), b2, 0, st>>>(c.glow.f(), w, outX, outY, outW, outH, c.packed.f());
        EG_CUDA_CHECK(cudaGetLastError());
        EG_CUDA_CHECK(c.staging.downloadRows(c.packed.f(), 3 * (size_t)outW * sizeof(float), outH, st,
                                             [&](int y0, int y1, const void* i) { glowOut(y0, y1, static_cast<const float*>(i)); }));
    }
    profiler().mark("download (GPU->host)");
    EG_CUDA_CHECK(cudaStreamSynchronize(st));
    profiler().end();
    return true;
}

} // namespace expoglow
