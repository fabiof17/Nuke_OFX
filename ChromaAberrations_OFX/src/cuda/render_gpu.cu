// ChromaAberrations_OFX - CUDA renderer (same pipeline as renderCpu, with cuFFT)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Per frame: upload the image, build the light of every wavelength over the padded domain
// (spectral conversion + inverse distortion + illumination). Then for each grid point: fill
// its weighted tile for all wavelengths at once, one batched forward transform, multiply by
// the PSF spectra and sum over the wavelengths into the three channels, one batched inverse
// transform, and add the tile back into the image. Download.

#include "render_gpu.h"

#include <chrono>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#if defined(_WIN32) && defined(CA_CUFFT_DLL)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstring>
#include <mutex>
#endif

#include "rendercommon.h"

namespace ca {

#define CA_CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fail(error, #call, cudaGetErrorString(e_)); return false; } } while (0)
#define CA_CUFFT(call) do { cufftResult r_ = (call); if (r_ != CUFFT_SUCCESS) { char b_[32]; std::snprintf(b_, sizeof(b_), "cuFFT error %d", (int)r_); fail(error, #call, b_); return false; } } while (0)

static void fail(std::string* error, const char* what, const char* why)
{
    if (error) {
        *error = std::string(what) + ": " + why;
    }
}

// ---------------------------------------------------------------------------- kernels

__global__ void kSource(PixelMath pm, const float* r, const float* g, const float* b,
                        const float* dispX, const float* dispY, const float* illum,
                        const float* lut, float br, float bg, float bb, int dw, int dh, float* f)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (i >= dw) || (j >= dh) ) {
        return;
    }
    f[(size_t)j * dw + i] = sourceValue(pm, i, j, r, g, b, dispX, dispY, illum, lut, br, bg, bb);
}

struct TileInfo
{
    int gx, gy, i0, i1, j0, j1, ox, oy;
};

// weighted light of one grid point, all wavelengths (blockIdx.z)
__global__ void kTileFill(PixelMath pm, const float* f, int dw, int dh, TileInfo t, int tw, int th, float* tiles)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, l = blockIdx.z;
    if ( (x >= tw) || (y >= th) ) {
        return;
    }
    const int i = t.ox + x, j = t.oy + y;
    float v = 0.f;
    if ( (i >= t.i0) && (i <= t.i1) && (j >= t.j0) && (j <= t.j1) ) {
        v = f[(size_t)l * dw * dh + (size_t)j * dw + i] * nodeWeight(pm, t.gx, t.gy, i, j);
    }
    tiles[(size_t)l * tw * th + (size_t)y * tw + x] = v;
}

// PSFs of one grid point, all wavelengths, centred at the origin with wrap-around
// wavelengths l0 + blockIdx.z of the bank
__global__ void kPlaceKernels(const float* bank, int G, int g, int S, int R, int tw, int th, int l0, float* dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, l = blockIdx.z;
    if ( (x >= S) || (y >= S) ) {
        return;
    }
    const size_t P = (size_t)S * S;
    const int xx = (x - R + tw) % tw, yy = (y - R + th) % th;
    dst[(size_t)l * tw * th + (size_t)yy * tw + xx] = bank[( (size_t)(l0 + l) * G + g ) * P + (size_t)y * S + x];
}

// acc_c = sum over the nl wavelengths of the batch of response_c(l) * T_l * E_l;
// resp[c * stride + l] (resp points at the first wavelength of the batch)
__global__ void kMacReduce(const cufftComplex* T, const cufftComplex* E, int nl, size_t n,
                           const float* resp, int stride, cufftComplex* acc)
{
    const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    float a0x = 0.f, a0y = 0.f, a1x = 0.f, a1y = 0.f, a2x = 0.f, a2y = 0.f;
    for (int l = 0; l < nl; ++l) {
        const cufftComplex t = T[(size_t)l * n + p], e = E[(size_t)l * n + p];
        const float re = t.x * e.x - t.y * e.y, im = t.x * e.y + t.y * e.x;
        const float w0 = resp[l], w1 = resp[stride + l], w2 = resp[2 * stride + l];
        a0x += w0 * re; a0y += w0 * im;
        a1x += w1 * re; a1y += w1 * im;
        a2x += w2 * re; a2y += w2 * im;
    }
    acc[p].x = a0x; acc[p].y = a0y;
    acc[n + p].x = a1x; acc[n + p].y = a1y;
    acc[2 * n + p].x = a2x; acc[2 * n + p].y = a2y;
}

// adds the three channel tiles (blockIdx.z) back into the image
__global__ void kOverlapAdd(const float* res, int tw, int th, int ox, int oy, int R, int W, int H, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if ( (x >= tw) || (y >= th) ) {
        return;
    }
    const int ix = ox + x - R, iy = oy + y - R;
    if ( (ix < 0) || (ix >= W) || (iy < 0) || (iy >= H) ) {
        return;
    }
    out[(size_t)c * W * H + (size_t)iy * W + ix] += res[(size_t)c * tw * th + (size_t)y * tw + x];
}

// no blur: sum of the wavelengths of the batch (padding R = 0), added to out
__global__ void kAccumulate(const float* f, int nl, size_t n, const float* resp, int stride, float* out)
{
    const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    float o0 = 0.f, o1 = 0.f, o2 = 0.f;
    for (int l = 0; l < nl; ++l) {
        const float v = f[(size_t)l * n + p];
        o0 += resp[l] * v;
        o1 += resp[stride + l] * v;
        o2 += resp[2 * stride + l] * v;
    }
    out[p] += o0;
    out[n + p] += o1;
    out[2 * n + p] += o2;
}

// ---------------------------------------------------------------------------- renderer

template <typename T>
static void freeDev(T*& p)
{
    if (p) {
        cudaFree(p);
        p = 0;
    }
}

struct GpuRenderer::Impl
{
    // plan
    bool hasPlan = false;
    LensPlan plan;             // host copy (tile layout, responses)
    PixelMath pm;
    int S = 1, R = 0, nl = 0, G = 1;
    float* dBank = 0;          // nl * G * S * S
    float* dDispX = 0;         // nl * mapW * mapH
    float* dDispY = 0;
    float* dIllum = 0;         // mapW * mapH
    float* dResp = 0;          // 3 * nl, response of each channel to each wavelength
    float* dLut = 0;           // nl * lutN * lutN, spectral table
    size_t mapSize = 0;
    std::string deviceName;

    // frame buffers (for the current image size)
    int W = 0, H = 0;
    TileLayout tl;
    float* dIn = 0;            // 3 planes W x H
    float* dOut = 0;           // 3 planes W x H
    float* dF = 0;             // nl x dw x dh
    float* dTiles = 0;         // nl x tw x th
    float* dKer = 0;           // nl x tw x th
    float* dRes = 0;           // 3 x tw x th
    cufftComplex* dT = 0;      // nl x cw x th
    cufftComplex* dE = 0;      // nl x cw x th
    cufftComplex* dAcc = 0;    // 3 x cw x th
    float* dRespScaled = 0;    // 3 * nl, with the FFT normalisation
    cufftHandle fwd = 0, fwdLast = 0, inv = 0;
    bool plans = false, planLast = false;
    int B = 1;                 // wavelengths per batch (all of them when the memory allows)
    int forceB = 0;            // tests: batch size imposed

    // page-locked host buffers (3 planes each)
    float* hIn = 0;
    float* hOut = 0;
    size_t hSize = 0;

    // PSF spectra, kept between frames when they fit: per grid point, nl x cw x th
    std::vector<cufftComplex*> cache;
    bool cacheValid = false;
    size_t cacheBudget = (size_t)-1;   // -1: a quarter of the free memory

    void freeFrame()
    {
        freeDev(dIn); freeDev(dOut); freeDev(dF); freeDev(dTiles); freeDev(dKer); freeDev(dRes);
        freeDev(dT); freeDev(dE); freeDev(dAcc); freeDev(dRespScaled);
        if (plans) {
            cufftDestroy(fwd);
            cufftDestroy(inv);
            plans = false;
        }
        if (planLast) {
            cufftDestroy(fwdLast);
            planLast = false;
        }
        freeCache();
        W = H = 0;
    }
    void freeCache()
    {
        for (cufftComplex*& p : cache) {
            freeDev(p);
        }
        cache.clear();
        cacheValid = false;
    }
    void freePlan()
    {
        freeDev(dBank); freeDev(dDispX); freeDev(dDispY); freeDev(dIllum); freeDev(dResp); freeDev(dLut);
        freeCache();
        hasPlan = false;
    }
};

GpuRenderer::GpuRenderer() : _d(new Impl) {}

GpuRenderer::~GpuRenderer()
{
    _d->freeFrame();
    _d->freePlan();
    if (_d->hIn) cudaFreeHost(_d->hIn);
    if (_d->hOut) cudaFreeHost(_d->hOut);
    delete _d;
}

float* GpuRenderer::inputBuffer(int width, int height)
{
    Impl& d = *_d;
    const size_t n = (size_t)3 * width * height;
    if (n != d.hSize) {
        if (d.hIn) cudaFreeHost(d.hIn);
        if (d.hOut) cudaFreeHost(d.hOut);
        d.hIn = d.hOut = 0;
        d.hSize = 0;
        if ( (cudaHostAlloc(&d.hIn, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) ||
             (cudaHostAlloc(&d.hOut, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) ) {
            cudaGetLastError();
            if (d.hIn) cudaFreeHost(d.hIn);
            d.hIn = d.hOut = 0;
            return 0;
        }
        d.hSize = n;
    }
    return d.hIn;
}

const float* GpuRenderer::outputBuffer() const
{
    return _d->hOut;
}

bool GpuRenderer::renderBuffers(int width, int height, RenderStats* stats, std::string* error)
{
    Impl& d = *_d;
    if ( !d.hIn || (d.hSize != (size_t)3 * width * height) ) {
        fail(error, "renderBuffers", "inputBuffer() was not called for this size");
        return false;
    }
    const size_t n = (size_t)width * height;
    const float* in[3] = { d.hIn, d.hIn + n, d.hIn + 2 * n };
    float* out[3] = { d.hOut, d.hOut + n, d.hOut + 2 * n };
    return renderPlanes(width, height, in, out, stats, error);
}

bool GpuRenderer::render(const ImageRGB& in, ImageRGB& out, RenderStats* stats, std::string* error)
{
    out.resize(in.width, in.height);
    const float* pin[3] = { in.c[0].data(), in.c[1].data(), in.c[2].data() };
    float* pout[3] = { out.c[0].data(), out.c[1].data(), out.c[2].data() };
    return renderPlanes(in.width, in.height, pin, pout, stats, error);
}

#if defined(_WIN32) && defined(CA_CUFFT_DLL)
// cuFFT is a DLL on Windows, delay-loaded by the plugin: load it from the folder of this module
// (the plugin bundle, or the folder of the program), then from the usual search path.
static bool loadCufft()
{
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, []() {
        HMODULE self = 0;
        wchar_t path[MAX_PATH * 2] = { 0 };
        if ( GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                (LPCWSTR)&loadCufft, &self) &&
             GetModuleFileNameW(self, path, MAX_PATH * 2) ) {
            std::wstring dir(path);
            const size_t slash = dir.find_last_of(L"\\/");
            if (slash != std::wstring::npos) {
                const char* dll = CA_CUFFT_DLL;
                std::wstring full = dir.substr(0, slash + 1) + std::wstring(dll, dll + strlen(dll));
                ok = LoadLibraryW( full.c_str() ) != 0;
            }
        }
        if (!ok) {
            ok = LoadLibraryA(CA_CUFFT_DLL) != 0;
        }
    });
    return ok;
}
#else
static bool loadCufft() { return true; }
#endif

bool GpuRenderer::available(std::string* name)
{
    if ( !loadCufft() ) {
        if (name) {
            *name = "cuFFT DLL not found";
        }
        return false;
    }
    int n = 0;
    if ( (cudaGetDeviceCount(&n) != cudaSuccess) || (n <= 0) ) {
        cudaGetLastError();
        if (name) {
            *name = "no CUDA device";
        }
        return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        cudaGetLastError();
        if (name) {
            *name = "no CUDA device";
        }
        return false;
    }
    if (name) {
        *name = prop.name;
    }
    return true;
}

void GpuRenderer::setWavelengthBatch(int n)
{
    _d->forceB = n;
    _d->freeFrame();
}

void GpuRenderer::setKernelCacheBudget(size_t bytes)
{
    _d->cacheBudget = bytes;
    _d->freeCache();
}

bool GpuRenderer::setPlan(const LensPlan& plan, std::string* error)
{
    Impl& d = *_d;
    d.freePlan();
    d.freeFrame();   // the tile layout depends on the plan
    d.plan = plan;
    d.pm = makePixelMath(plan);
    d.S = plan.S;
    d.R = plan.R;
    d.nl = plan.wavelengthCount();
    d.G = plan.nodeCount();
    const size_t P = (size_t)d.S * d.S;
    d.mapSize = (size_t)plan.mapW * plan.mapH;
    std::vector<float> bank( (size_t)d.nl * d.G * P ), dx(d.mapSize * d.nl), dy(d.mapSize * d.nl), resp(3 * d.nl);
    for (int l = 0; l < d.nl; ++l) {
        std::copy(plan.bankKernels[l].begin(), plan.bankKernels[l].end(), bank.begin() + (size_t)l * d.G * P);
        std::copy(plan.dispX[l].begin(), plan.dispX[l].end(), dx.begin() + l * d.mapSize);
        std::copy(plan.dispY[l].begin(), plan.dispY[l].end(), dy.begin() + l * d.mapSize);
        for (int c = 0; c < 3; ++c) {
            resp[c * d.nl + l] = (float)plan.spectral.response[c][l];
        }
    }
    CA_CUDA( cudaSetDevice(0) );
    available(&d.deviceName);
    CA_CUDA( cudaMalloc(&d.dBank, bank.size() * sizeof(float)) );
    CA_CUDA( cudaMalloc(&d.dDispX, dx.size() * sizeof(float)) );
    CA_CUDA( cudaMalloc(&d.dDispY, dy.size() * sizeof(float)) );
    CA_CUDA( cudaMalloc(&d.dIllum, d.mapSize * sizeof(float)) );
    CA_CUDA( cudaMalloc(&d.dResp, resp.size() * sizeof(float)) );
    CA_CUDA( cudaMemcpy(d.dBank, bank.data(), bank.size() * sizeof(float), cudaMemcpyHostToDevice) );
    CA_CUDA( cudaMemcpy(d.dDispX, dx.data(), dx.size() * sizeof(float), cudaMemcpyHostToDevice) );
    CA_CUDA( cudaMemcpy(d.dDispY, dy.data(), dy.size() * sizeof(float), cudaMemcpyHostToDevice) );
    CA_CUDA( cudaMemcpy(d.dIllum, plan.illum.data(), d.mapSize * sizeof(float), cudaMemcpyHostToDevice) );
    CA_CUDA( cudaMemcpy(d.dResp, resp.data(), resp.size() * sizeof(float), cudaMemcpyHostToDevice) );
    if ( !plan.spectral.lut.empty() ) {
        CA_CUDA( cudaMalloc(&d.dLut, plan.spectral.lut.size() * sizeof(float)) );
        CA_CUDA( cudaMemcpy(d.dLut, plan.spectral.lut.data(), plan.spectral.lut.size() * sizeof(float), cudaMemcpyHostToDevice) );
    }
    d.hasPlan = true;
    return true;
}

bool GpuRenderer::renderPlanes(int W, int H, const float* const in[3], float* const out[3], RenderStats* stats, std::string* error)
{
    typedef std::chrono::steady_clock Clock;
    const auto t0 = Clock::now();
    Impl& d = *_d;
    if (!d.hasPlan) {
        fail(error, "render", "no lens plan");
        return false;
    }
    if ( (W != d.pm.width) || (H != d.pm.height) ) {
        fail(error, "render", "image size differs from the lens plan");
        return false;
    }
    const bool blur = d.S > 1;
    const int dw = W + 2 * d.R, dh = H + 2 * d.R;
    const size_t nImg = (size_t)W * H, nDom = (size_t)dw * dh;

    // ---- buffers for this size
    if ( (W != d.W) || (H != d.H) ) {
        d.freeFrame();
        d.tl = makeTileLayout(d.plan);
        const int tw0 = d.tl.tw, th0 = d.tl.th;
        const size_t nReal0 = (size_t)tw0 * th0, nCplx0 = (size_t)(tw0 / 2 + 1) * th0;
        // wavelengths per batch: as many as fit in half of the free memory (Final in 4K)
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        const size_t perL = nDom * sizeof(float) + (blur ? 2 * nReal0 * sizeof(float) + 2 * nCplx0 * sizeof(cufftComplex) : 0);
        const size_t fixed = 6 * nImg * sizeof(float) + (blur ? 3 * nReal0 * sizeof(float) + 3 * nCplx0 * sizeof(cufftComplex) : 0);
        const size_t budget = freeMem / 2 > fixed ? freeMem / 2 - fixed : 0;
        d.B = (int)std::max<size_t>( 1, std::min<size_t>( d.nl, budget / std::max<size_t>(1, perL) ) );
        if (d.forceB > 0) d.B = std::min(d.forceB, d.nl);
        CA_CUDA( cudaMalloc(&d.dIn, 3 * nImg * sizeof(float)) );
        CA_CUDA( cudaMalloc(&d.dOut, 3 * nImg * sizeof(float)) );
        CA_CUDA( cudaMalloc(&d.dF, (size_t)d.B * nDom * sizeof(float)) );
        if (blur) {
            const int tw = d.tl.tw, th = d.tl.th, cw = tw / 2 + 1;
            const size_t nReal = (size_t)tw * th, nCplx = (size_t)cw * th;
            CA_CUDA( cudaMalloc(&d.dTiles, (size_t)d.B * nReal * sizeof(float)) );
            CA_CUDA( cudaMalloc(&d.dKer, (size_t)d.B * nReal * sizeof(float)) );
            CA_CUDA( cudaMalloc(&d.dRes, 3 * nReal * sizeof(float)) );
            CA_CUDA( cudaMalloc(&d.dT, (size_t)d.B * nCplx * sizeof(cufftComplex)) );
            CA_CUDA( cudaMalloc(&d.dE, (size_t)d.B * nCplx * sizeof(cufftComplex)) );
            CA_CUDA( cudaMalloc(&d.dAcc, 3 * nCplx * sizeof(cufftComplex)) );
            int n[2] = { th, tw };
            CA_CUFFT( cufftPlanMany(&d.fwd, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, d.B) );
            CA_CUFFT( cufftPlanMany(&d.inv, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, 3) );
            d.plans = true;
            const int rem = d.nl % d.B;
            if (rem) {
                CA_CUFFT( cufftPlanMany(&d.fwdLast, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, rem) );
                d.planLast = true;
            }
        }
        // responses with the FFT normalisation folded in
        std::vector<float> rs(3 * d.nl);
        const float norm = blur ? 1.f / ( (float)d.tl.tw * (float)d.tl.th ) : 1.f;
        for (int c = 0; c < 3; ++c) {
            for (int l = 0; l < d.nl; ++l) {
                rs[c * d.nl + l] = (float)d.plan.spectral.response[c][l] * norm;
            }
        }
        CA_CUDA( cudaMalloc(&d.dRespScaled, rs.size() * sizeof(float)) );
        CA_CUDA( cudaMemcpy(d.dRespScaled, rs.data(), rs.size() * sizeof(float), cudaMemcpyHostToDevice) );
        d.W = W;
        d.H = H;
    }
    const int tw = d.tl.tw, th = d.tl.th, cw = tw / 2 + 1;
    const size_t nReal = (size_t)tw * th, nCplx = (size_t)cw * th;

    // ---- upload
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0);
    for (int c = 0; c < 3; ++c) {
        CA_CUDA( cudaMemcpy(d.dIn + c * nImg, in[c], nImg * sizeof(float), cudaMemcpyHostToDevice) );
    }
    cudaEventRecord(e1);

    const dim3 blk(32, 8);
    const bool warp = d.plan.settings.distortion, vig = d.plan.settings.vignetting;
    auto fwdPlan = [&](int nb) { return nb == d.B ? d.fwd : d.fwdLast; };
    bool useCache = false;
    if (blur) {
        // ---- PSF spectra: kept on the GPU when they fit in the budget
        const size_t perNode = (size_t)d.nl * nCplx * sizeof(cufftComplex);
        if (d.cacheValid) {
            useCache = true;
        } else {
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            const size_t budget = d.cacheBudget == (size_t)-1 ? freeMem / 4 : d.cacheBudget;
            if ( (budget > 0) && ((size_t)d.G * perNode <= budget) ) {
                d.cache.assign(d.G, (cufftComplex*)0);
                bool ok = true;
                for (int g = 0; g < d.G && ok; ++g) {
                    ok = cudaMalloc(&d.cache[g], perNode) == cudaSuccess;
                }
                if (ok) {
                    for (int g = 0; g < d.G; ++g) {
                        for (int b0 = 0; b0 < d.nl; b0 += d.B) {
                            const int nb = std::min(d.B, d.nl - b0);
                            CA_CUDA( cudaMemset(d.dKer, 0, (size_t)nb * nReal * sizeof(float)) );
                            kPlaceKernels<<<dim3( (d.S + 15) / 16, (d.S + 15) / 16, nb ), dim3(16, 16)>>>(d.dBank, d.G, g, d.S, d.R, tw, th, b0, d.dKer);
                            CA_CUFFT( cufftExecR2C(fwdPlan(nb), d.dKer, d.cache[g] + (size_t)b0 * nCplx) );
                        }
                    }
                    d.cacheValid = true;
                    useCache = true;
                } else {
                    cudaGetLastError();
                    d.freeCache();
                }
            }
        }
    }
    CA_CUDA( cudaMemset(d.dOut, 0, 3 * nImg * sizeof(float)) );

    // ---- wavelengths by batches
    for (int b0 = 0; b0 < d.nl; b0 += d.B) {
        const int nb = std::min(d.B, d.nl - b0);
        // light of the batch's wavelengths over the padded domain
        for (int i = 0; i < nb; ++i) {
            const int l = b0 + i;
            kSource<<<dim3( (dw + 31) / 32, (dh + 7) / 8 ), blk>>>(d.pm, d.dIn, d.dIn + nImg, d.dIn + 2 * nImg,
                warp ? d.dDispX + l * d.mapSize : 0, warp ? d.dDispY + l * d.mapSize : 0, vig ? d.dIllum : 0,
                d.dLut ? d.dLut + (size_t)l * d.plan.spectral.lutN * d.plan.spectral.lutN : 0,
                (float)d.plan.spectral.basis[0][l], (float)d.plan.spectral.basis[1][l], (float)d.plan.spectral.basis[2][l],
                dw, dh, d.dF + (size_t)i * nDom);
        }
        if (!blur) {
            kAccumulate<<<(unsigned)( (nImg + 255) / 256 ), 256>>>(d.dF, nb, nImg, d.dRespScaled + b0, d.nl, d.dOut);
            continue;
        }
        // grid points: weighted tiles, convolution, sum over the batch, back into the image
        const dim3 grdT( (tw + 31) / 32, (th + 7) / 8, nb );
        const dim3 grdO( (tw + 31) / 32, (th + 7) / 8, 3 );
        for (int g = 0; g < d.G; ++g) {
            TileInfo t;
            t.gx = g % d.plan.gridX;
            t.gy = g / d.plan.gridX;
            t.i0 = d.tl.i0[g]; t.i1 = d.tl.i1[g]; t.j0 = d.tl.j0[g]; t.j1 = d.tl.j1[g];
            t.ox = d.tl.ox[g]; t.oy = d.tl.oy[g];
            kTileFill<<<grdT, blk>>>(d.pm, d.dF, dw, dh, t, tw, th, d.dTiles);
            CA_CUFFT( cufftExecR2C(fwdPlan(nb), d.dTiles, d.dT) );
            const cufftComplex* E = 0;
            if (useCache) {
                E = d.cache[g] + (size_t)b0 * nCplx;
            } else {
                CA_CUDA( cudaMemsetAsync(d.dKer, 0, (size_t)nb * nReal * sizeof(float)) );
                kPlaceKernels<<<dim3( (d.S + 15) / 16, (d.S + 15) / 16, nb ), dim3(16, 16)>>>(d.dBank, d.G, g, d.S, d.R, tw, th, b0, d.dKer);
                CA_CUFFT( cufftExecR2C(fwdPlan(nb), d.dKer, d.dE) );
                E = d.dE;
            }
            kMacReduce<<<(unsigned)( (nCplx + 255) / 256 ), 256>>>(d.dT, E, nb, nCplx, d.dRespScaled + b0, d.nl, d.dAcc);
            CA_CUFFT( cufftExecC2R(d.inv, d.dAcc, d.dRes) );
            kOverlapAdd<<<grdO, blk>>>(d.dRes, tw, th, t.ox, t.oy, d.R, W, H, d.dOut);
        }
    }
    CA_CUDA( cudaGetLastError() );
    cudaEventRecord(e2);

    // ---- download
    for (int c = 0; c < 3; ++c) {
        CA_CUDA( cudaMemcpy(out[c], d.dOut + c * nImg, nImg * sizeof(float), cudaMemcpyDeviceToHost) );
    }
    cudaEventRecord(e3);
    CA_CUDA( cudaEventSynchronize(e3) );
    float msUp = 0.f, msRun = 0.f, msDown = 0.f;
    cudaEventElapsedTime(&msUp, e0, e1);
    cudaEventElapsedTime(&msRun, e1, e2);
    cudaEventElapsedTime(&msDown, e2, e3);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    if (stats) {
        stats->fftWidth = blur ? tw : 0;
        stats->fftHeight = blur ? th : 0;
        stats->convolutions = blur ? d.G * d.nl : 0;
        stats->seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        stats->sourceSeconds = 0.;
        stats->fftSeconds = msRun * 1e-3;
        stats->transferSeconds = (msUp + msDown) * 1e-3;
        stats->kernelCache = useCache;
        stats->batches = (d.nl + d.B - 1) / d.B;
        stats->device = d.deviceName;
    }
    return true;
}

} // namespace ca
