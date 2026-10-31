/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX, a GPU-accelerated derivative of the
 * openfx-misc CornerPin plugin <https://github.com/NatronGitHub/openfx-misc>.
 * (C) 2026 CornerPinEx_OFX contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * CUDA render engine of CornerPinEx_OFX.
 *
 * The OFX host (Nuke) gives images in CPU memory, so each render:
 *  1. uploads the source (and mask) to the GPU and converts it to float,
 *  2. builds a mip pyramid when an EWA filter is used,
 *  3. renders every output pixel of the render window in one kernel
 *     (filtering + motion blur + mask/mix), using the code of cpSampler.h,
 *  4. converts to the output depth and downloads the result.
 *
 * Performance: the plugin asks the host for full frames (no tiles), so there is one
 * upload/download per frame. GPU resources are kept between renders in a small pool of
 * "slots" (stream, device buffers that only grow, pinned staging buffers): host<->device
 * transfers go through double-buffered pinned memory, at full PCIe speed.
 * Set CORNERPINEX_OFX_PROFILE=1 to log the time of each step to
 * %TEMP%/CornerPinEx_OFX_profile.log (or /tmp/CornerPinEx_OFX_profile.log).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "cpEngine.h"

namespace cpg {

namespace {

////////////////////////////////////////////////////////////////////////////////
// kernels

template <class PIX, int NC>
__global__ void kToFloat(const PIX* __restrict__ in, float* __restrict__ out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (x >= w) || (y >= h) ) {
        return;
    }
    size_t o = ( (size_t)y * w + x ) * NC;
    for (int c = 0; c < NC; ++c) {
        out[o + c] = (float)in[o + c];
    }
}

template <class PIX>
__global__ void kMaskToFloat(const PIX* __restrict__ in, float* __restrict__ out, int w, int h, float scale, bool invert)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (x >= w) || (y >= h) ) {
        return;
    }
    size_t o = (size_t)y * w + x;
    float m = (float)in[o] * scale;
    out[o] = invert ? 1.f - m : m;
}

template <int NC>
__global__ void kDownsample(FImage src, float* dst, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (x >= w) || (y >= h) ) {
        return;
    }
    downsamplePixel<NC>(src, dst, (long long)w * NC, x, y);
}

struct KernelArgs
{
    Pyramid pyr;
    SampleParams sp;
    const Mat3* H;
    const float* alpha;
    int count;
    float motionblur;
    float maxValue;
    // render window
    int rx1, ry1, rw, rh;
    // mask (float, already inverted if needed), may be NULL
    const float* mask;
    int mx1, my1, mw, mh;
    bool masked, doMask, maskInvert;
    float mix;
    bool hasSrc;
};

template <int NC>
__global__ void kRender(KernelArgs a, float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (i >= a.rw) || (j >= a.rh) ) {
        return;
    }
    const int x = a.rx1 + i;
    const int y = a.ry1 + j;
    float tmp[NC];
    if (a.hasSrc) {
        BlurParams bp;
        bp.H = a.H;
        bp.alpha = a.alpha;
        bp.count = a.count;
        bp.motionblur = a.motionblur;
        bp.maxValue = a.maxValue;
        renderPixel<NC>(a.pyr, x, y, a.sp, bp, tmp);
    } else {
        for (int c = 0; c < NC; ++c) {
            tmp[c] = 0.f;
        }
    }

    float alphaMix = 1.f;
    if (a.masked) {
        float maskScale = 1.f;
        if (a.doMask) {
            int mi = x - a.mx1, mj = y - a.my1;
            if ( a.mask && (mi >= 0) && (mj >= 0) && (mi < a.mw) && (mj < a.mh) ) {
                maskScale = a.mask[(size_t)mj * a.mw + mi];
            } else {
                maskScale = a.maskInvert ? 1.f : 0.f;
            }
        }
        alphaMix = maskScale * a.mix;
    }
    float* o = out + ( (size_t)j * a.rw + i ) * NC;
    if (alphaMix == 1.f) {
        for (int c = 0; c < NC; ++c) {
            o[c] = tmp[c];
        }
        return;
    }
    // untransformed source pixel for mixing (NULL if outside the source bounds)
    const FImage& l0 = a.pyr.level[0];
    int si = x - a.pyr.x1, sj = y - a.pyr.y1;
    const float* srcPix = NULL;
    if ( a.hasSrc && (si >= 0) && (sj >= 0) && (si < l0.width) && (sj < l0.height) ) {
        srcPix = l0.data + (long long)sj * l0.stride + (long long)si * NC;
    }
    maskMix<NC>(tmp, srcPix, alphaMix, o);
}

template <class PIX, int NC>
__global__ void kFromFloat(const float* __restrict__ in, PIX* __restrict__ out, int w, int h, float maxValue)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( (x >= w) || (y >= h) ) {
        return;
    }
    size_t o = ( (size_t)y * w + x ) * NC;
    for (int c = 0; c < NC; ++c) {
        float v = in[o + c];
        v = fminf(fmaxf(v, 0.f), maxValue) + 0.5f;
        out[o + c] = (PIX)v;
    }
}

inline dim3 grid2d(int w, int h, dim3 block)
{
    return dim3( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y );
}

size_t bytesPerComp(PixelDepth d)
{
    return d == eDepthUByte ? 1 : (d == eDepthUShort ? 2 : 4);
}

template <int NC>
void launchToFloat(PixelDepth depth, const void* in, float* out, int w, int h, cudaStream_t s)
{
    dim3 b(32, 8);
    if (depth == eDepthUByte) {
        kToFloat<unsigned char, NC><<<grid2d(w, h, b), b, 0, s>>>( (const unsigned char*)in, out, w, h );
    } else if (depth == eDepthUShort) {
        kToFloat<unsigned short, NC><<<grid2d(w, h, b), b, 0, s>>>( (const unsigned short*)in, out, w, h );
    }
}

template <int NC>
void launchFromFloat(PixelDepth depth, const float* in, void* out, int w, int h, float maxValue, cudaStream_t s)
{
    dim3 b(32, 8);
    if (depth == eDepthUByte) {
        kFromFloat<unsigned char, NC><<<grid2d(w, h, b), b, 0, s>>>( in, (unsigned char*)out, w, h, maxValue );
    } else if (depth == eDepthUShort) {
        kFromFloat<unsigned short, NC><<<grid2d(w, h, b), b, 0, s>>>( in, (unsigned short*)out, w, h, maxValue );
    }
}

template <int NC>
void launchDownsample(const FImage& src, float* dst, int w, int h, cudaStream_t s)
{
    dim3 b(32, 8);
    kDownsample<NC><<<grid2d(w, h, b), b, 0, s>>>(src, dst, w, h);
}

template <int NC>
void launchRender(const KernelArgs& a, float* out, cudaStream_t s)
{
    // 16x8 blocks: motion blur and EWA make threads diverge, smaller blocks help
    dim3 b(16, 8);
    kRender<NC><<<grid2d(a.rw, a.rh, b), b, 0, s>>>(a, out);
}

#define CPG_DISPATCH_NC(nc, f, ...) \
    switch (nc) { \
    case 1: f<1>(__VA_ARGS__); break; \
    case 2: f<2>(__VA_ARGS__); break; \
    case 3: f<3>(__VA_ARGS__); break; \
    default: f<4>(__VA_ARGS__); break; \
    }

std::string errString(const char* what, cudaError_t e)
{
    std::ostringstream os;
    os << "CUDA " << what << ": " << cudaGetErrorString(e);
    return os.str();
}


////////////////////////////////////////////////////////////////////////////////
// GPU slots: persistent resources reused from one render to the next.

enum BufferId
{
    eBufSrcRaw = 0, eBufLevels, eBufH, eBufAlpha, eBufMaskRaw, eBufMask, eBufOutF, eBufOutRaw, eBufCount
};

// Per pinned staging buffer (two per slot). Small enough for the CPU copy of one chunk to overlap
// with the DMA of the previous one (measured: 32 MB chunks left half of the upload unoverlapped).
#define CPG_STAGING_BYTES (8u << 20)
#define CPG_PARALLEL_COPY_MIN (1u << 20) // copies smaller than this are done by one thread

// Number of threads for host-side copies (Nuke image <-> pinned buffer).
// CORNERPINEX_OFX_COPY_THREADS overrides it (1 = no parallel copy).
int copyThreadCount()
{
    static int n = 0;
    if (n == 0) {
        const char* v = std::getenv("CORNERPINEX_OFX_COPY_THREADS");
        int t = v ? std::atoi(v) : 0;
        if (t <= 0) {
            unsigned hc = std::thread::hardware_concurrency();
            t = hc >= 12 ? 6 : (hc >= 4 ? (int)hc / 2 : 1);
        }
        n = std::max(1, std::min(t, 16));
    }
    return n;
}

// How host <-> device transfers are done. CORNERPINEX_OFX_TRANSFER selects it:
//  "staged"   (default) copy through two pinned buffers, CPU copy split over several threads
//  "direct"   cudaMemcpy2D from/to the host image (the driver does its own staging)
//  "register" the host image is page-locked for the transfer (cudaHostRegister): no CPU copy
enum TransferMode { eTransferStaged = 0, eTransferDirect, eTransferRegister };

TransferMode transferMode()
{
    static int m = -1;
    if (m < 0) {
        const char* v = std::getenv("CORNERPINEX_OFX_TRANSFER");
        std::string s = v ? v : "";
        m = (s == "direct") ? eTransferDirect : (s == "register") ? eTransferRegister : eTransferStaged;
    }
    return (TransferMode)m;
}

const char* transferModeName(TransferMode m)
{
    return m == eTransferDirect ? "direct" : (m == eTransferRegister ? "register" : "staged");
}

// Persistent threads for the host-side copies: a single thread is limited by memory latency
// and by the page faults of freshly allocated host memory, not by bandwidth.
// The threads are created once per GPU slot and live as long as the process.
// Windows (WDDM) batches GPU commands and may submit them only at the next synchronization:
// an async copy queued here could then wait until the CPU has finished copying the next chunk,
// and the copies would not overlap. cudaStreamQuery() forces the submission without waiting.
inline void flushStream(cudaStream_t s)
{
    if (cudaStreamQuery(s) == cudaErrorNotReady) {
        cudaGetLastError(); // not an error: work is in flight
    }
}

class CopyPool
{
public:
    void copy(unsigned char* dst, long long dstPitch, const unsigned char* src, long long srcPitch,
              size_t rowBytes, int rows)
    {
        const int nt = copyThreadCount();
        if ( (rowBytes * (size_t)rows < CPG_PARALLEL_COPY_MIN) || (nt <= 1) || (rows < 2 * nt) ) {
            copyRows(dst, dstPitch, src, srcPitch, rowBytes, 0, rows);
            return;
        }
        std::unique_lock<std::mutex> lock(_m);
        startThreads(nt - 1); // under the lock: new threads start at the current generation
        _dst = dst; _dstPitch = dstPitch; _src = src; _srcPitch = srcPitch;
        _rowBytes = rowBytes; _rows = rows;
        _parts = (int)_threads.size() + 1;
        _remaining = (int)_threads.size();
        ++_generation;
        lock.unlock();
        _cv.notify_all();
        doPart(0);
        lock.lock();
        _cvDone.wait(lock, [this] { return _remaining == 0; });
    }

private:
    static void copyRows(unsigned char* dst, long long dstPitch, const unsigned char* src, long long srcPitch,
                         size_t rowBytes, int r0, int r1)
    {
        if ( (dstPitch == (long long)rowBytes) && (srcPitch == (long long)rowBytes) ) {
            std::memcpy(dst + (long long)r0 * dstPitch, src + (long long)r0 * srcPitch, (size_t)(r1 - r0) * rowBytes);
        } else {
            for (int r = r0; r < r1; ++r) {
                std::memcpy(dst + (long long)r * dstPitch, src + (long long)r * srcPitch, rowBytes);
            }
        }
    }

    void doPart(int i)
    {
        const int r0 = (int)( (long long)_rows * i / _parts );
        const int r1 = (int)( (long long)_rows * (i + 1) / _parts );
        copyRows(_dst, _dstPitch, _src, _srcPitch, _rowBytes, r0, r1);
    }

    void startThreads(int n)
    {
        if ( (int)_threads.size() >= n ) {
            return;
        }
        for (int i = (int)_threads.size(); i < n; ++i) {
            try {
                _threads.emplace_back(&CopyPool::worker, this, i + 1, _generation);
                _threads.back().detach();
            } catch (...) {
                break; // fewer threads
            }
        }
    }

    void worker(int index, unsigned long long seen)
    {
        std::unique_lock<std::mutex> lock(_m);
        for (;;) {
            _cv.wait(lock, [&] { return _generation != seen; });
            seen = _generation;
            const bool mine = index < _parts;
            lock.unlock();
            if (mine) {
                doPart(index);
            }
            lock.lock();
            if (mine && --_remaining == 0) {
                _cvDone.notify_one();
            }
        }
    }

    std::vector<std::thread> _threads;
    std::mutex _m;
    std::condition_variable _cv, _cvDone;
    unsigned long long _generation = 0;
    int _remaining = 0;
    int _parts = 1;
    unsigned char* _dst = NULL;
    const unsigned char* _src = NULL;
    long long _dstPitch = 0, _srcPitch = 0;
    size_t _rowBytes = 0;
    int _rows = 0;
};

#define CPG_NUM_SLOTS 2

struct GpuSlot
{
    int device = -1;
    cudaStream_t stream = 0;
    void* buf[eBufCount] = {};
    size_t bufSize[eBufCount] = {};
    unsigned char* pinned[2] = { NULL, NULL };
    size_t pinnedSize = 0;
    cudaEvent_t ev[2] = { 0, 0 };
    bool inUse = false;
    bool trimRequested = false; // an allocation failed: free all buffers when the render is over
    double hostCopyMs = 0.;     // time spent in host-side copies or page-locking (profiling)
    CopyPool* copier = NULL;    // created on first use, never destroyed (see CopyPool)

    void timedCopy(unsigned char* dst, long long dstPitch, const unsigned char* src, long long srcPitch,
                   size_t rowBytes, int rows)
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!copier) {
            copier = new CopyPool;
        }
        copier->copy(dst, dstPitch, src, srcPitch, rowBytes, rows);
        hostCopyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // "register" mode: page-lock the host image span for the duration of one transfer
    bool registerHost(const void* p, long long pitch, size_t rowBytes, int rows)
    {
        auto t0 = std::chrono::steady_clock::now();
        const size_t span = (size_t)(rows - 1) * (size_t)pitch + rowBytes;
        cudaError_t e = cudaHostRegister(const_cast<void*>(p), span, cudaHostRegisterDefault);
        hostCopyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (e != cudaSuccess) {
            cudaGetLastError(); // e.g. already registered by another render: use the staged path
            return false;
        }
        return true;
    }

    void unregisterHost(const void* p)
    {
        auto t0 = std::chrono::steady_clock::now();
        cudaHostUnregister( const_cast<void*>(p) );
        cudaGetLastError();
        hostCopyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    void releaseBuffers()
    {
        for (int i = 0; i < eBufCount; ++i) {
            if (buf[i]) {
                cudaFree(buf[i]);
            }
            buf[i] = NULL;
            bufSize[i] = 0;
        }
    }

    void releaseAll()
    {
        if (device < 0) {
            return;
        }
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        releaseBuffers();
        for (int i = 0; i < 2; ++i) {
            if (pinned[i]) {
                cudaFreeHost(pinned[i]);
            }
            pinned[i] = NULL;
            if (ev[i]) {
                cudaEventDestroy(ev[i]);
            }
            ev[i] = 0;
        }
        pinnedSize = 0;
        if (stream) {
            cudaStreamDestroy(stream);
        }
        stream = 0;
        device = -1;
        cudaGetLastError();
    }

    // make the slot usable on 'dev' (creates the stream, events and staging buffers once)
    cudaError_t init(int dev)
    {
        if (device != dev) {
            releaseAll();
            device = dev;
        }
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) {
            return e;
        }
        if (!stream) {
            e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (e != cudaSuccess) {
                return e;
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (!ev[i]) {
                e = cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
                if (e != cudaSuccess) {
                    return e;
                }
            }
        }
        return ensurePinned(CPG_STAGING_BYTES);
    }

    cudaError_t ensurePinned(size_t bytes)
    {
        if (pinnedSize >= bytes) {
            return cudaSuccess;
        }
        cudaStreamSynchronize(stream);
        for (int i = 0; i < 2; ++i) {
            if (pinned[i]) {
                cudaFreeHost(pinned[i]);
                pinned[i] = NULL;
            }
        }
        pinnedSize = 0;
        for (int i = 0; i < 2; ++i) {
            cudaError_t e = cudaHostAlloc( (void**)&pinned[i], bytes, cudaHostAllocDefault );
            if (e != cudaSuccess) {
                return e;
            }
        }
        pinnedSize = bytes;
        return cudaSuccess;
    }

    // device buffer that only grows (+12.5% headroom to avoid reallocating for small size changes)
    void* get(BufferId id, size_t bytes, cudaError_t* err)
    {
        bytes = std::max<size_t>(bytes, 256);
        if (bufSize[id] >= bytes) {
            return buf[id];
        }
        cudaStreamSynchronize(stream);
        if (buf[id]) {
            cudaFree(buf[id]);
            buf[id] = NULL;
            bufSize[id] = 0;
        }
        size_t want = bytes + bytes / 8;
        cudaError_t e = cudaMalloc(&buf[id], want);
        if (e != cudaSuccess) {
            cudaGetLastError();
            want = bytes;
            e = cudaMalloc(&buf[id], want);
        }
        if (e != cudaSuccess) {
            // out of memory: this render falls back to the CPU. The other buffers may still be in
            // use by this render, so they are freed only when it is over (see SlotGuard).
            cudaGetLastError();
            buf[id] = NULL;
            trimRequested = true;
            *err = e;
            return NULL;
        }
        bufSize[id] = want;
        return buf[id];
    }

    // host (pageable, any pitch) -> device (tight rows), through the two pinned buffers:
    // the CPU copy of chunk k overlaps with the DMA of chunk k-1.
    cudaError_t upload(void* dDst, const void* hSrc, long long hPitch, size_t rowBytes, int rows)
    {
        if ( (rows <= 0) || (rowBytes == 0) ) {
            return cudaSuccess;
        }
        const TransferMode mode = transferMode();
        if (mode == eTransferDirect) {
            return cudaMemcpy2DAsync(dDst, rowBytes, hSrc, (size_t)hPitch, rowBytes, rows, cudaMemcpyHostToDevice, stream);
        }
        if ( (mode == eTransferRegister) && registerHost(hSrc, hPitch, rowBytes, rows) ) {
            cudaError_t e = cudaMemcpy2DAsync(dDst, rowBytes, hSrc, (size_t)hPitch, rowBytes, rows, cudaMemcpyHostToDevice, stream);
            cudaError_t e2 = cudaStreamSynchronize(stream);
            unregisterHost(hSrc);
            return e != cudaSuccess ? e : e2;
        }
        cudaError_t e = ensurePinned( std::max<size_t>(CPG_STAGING_BYTES, rowBytes) );
        if (e != cudaSuccess) {
            return e;
        }
        const int rowsPerChunk = (int)std::max<size_t>(1, pinnedSize / rowBytes);
        int k = 0;
        for (int r0 = 0; r0 < rows; r0 += rowsPerChunk, ++k) {
            const int n = std::min(rowsPerChunk, rows - r0);
            const int b = k & 1;
            e = cudaEventSynchronize(ev[b]); // the previous DMA from this buffer is finished
            if (e != cudaSuccess) {
                return e;
            }
            unsigned char* p = pinned[b];
            const unsigned char* s = (const unsigned char*)hSrc + (long long)r0 * hPitch;
            timedCopy(p, (long long)rowBytes, s, hPitch, rowBytes, n);
            e = cudaMemcpyAsync( (unsigned char*)dDst + (size_t)r0 * rowBytes, p, (size_t)n * rowBytes,
                                 cudaMemcpyHostToDevice, stream );
            if (e != cudaSuccess) {
                return e;
            }
            e = cudaEventRecord(ev[b], stream);
            if (e != cudaSuccess) {
                return e;
            }
            flushStream(stream); // start the DMA now, while the CPU copies the next chunk
        }
        return cudaSuccess;
    }

    // device (tight rows) -> host (any pitch), through the pinned buffers
    cudaError_t download(void* hDst, long long hPitch, const void* dSrc, size_t rowBytes, int rows)
    {
        if ( (rows <= 0) || (rowBytes == 0) ) {
            return cudaSuccess;
        }
        const TransferMode mode = transferMode();
        if (mode == eTransferDirect) {
            return cudaMemcpy2DAsync(hDst, (size_t)hPitch, dSrc, rowBytes, rowBytes, rows, cudaMemcpyDeviceToHost, stream);
        }
        if ( (mode == eTransferRegister) && registerHost(hDst, hPitch, rowBytes, rows) ) {
            cudaError_t e = cudaMemcpy2DAsync(hDst, (size_t)hPitch, dSrc, rowBytes, rowBytes, rows, cudaMemcpyDeviceToHost, stream);
            cudaError_t e2 = cudaStreamSynchronize(stream);
            unregisterHost(hDst);
            return e != cudaSuccess ? e : e2;
        }
        cudaError_t e = ensurePinned( std::max<size_t>(CPG_STAGING_BYTES, rowBytes) );
        if (e != cudaSuccess) {
            return e;
        }
        const int rowsPerChunk = (int)std::max<size_t>(1, pinnedSize / rowBytes);
        const int nChunks = (rows + rowsPerChunk - 1) / rowsPerChunk;
        for (int k = 0; k <= nChunks; ++k) {
            if (k < nChunks) {
                const int r0 = k * rowsPerChunk;
                const int n = std::min(rowsPerChunk, rows - r0);
                e = cudaMemcpyAsync( pinned[k & 1], (const unsigned char*)dSrc + (size_t)r0 * rowBytes,
                                     (size_t)n * rowBytes, cudaMemcpyDeviceToHost, stream );
                if (e != cudaSuccess) {
                    return e;
                }
                e = cudaEventRecord(ev[k & 1], stream);
                if (e != cudaSuccess) {
                    return e;
                }
                flushStream(stream); // start the DMA now, while the CPU copies the previous chunk out
            }
            if (k > 0) {
                // copy out the previous chunk while the current one is transferred
                const int pk = k - 1;
                const int r0 = pk * rowsPerChunk;
                const int n = std::min(rowsPerChunk, rows - r0);
                e = cudaEventSynchronize(ev[pk & 1]);
                if (e != cudaSuccess) {
                    return e;
                }
                const unsigned char* p = pinned[pk & 1];
                unsigned char* d = (unsigned char*)hDst + (long long)r0 * hPitch;
                timedCopy(d, hPitch, p, (long long)rowBytes, rowBytes, n);
            }
        }
        return cudaSuccess;
    }
};

class SlotPool
{
public:
    GpuSlot* acquire()
    {
        std::unique_lock<std::mutex> lock(_m);
        for (;;) {
            for (int i = 0; i < CPG_NUM_SLOTS; ++i) {
                if (!_slots[i].inUse) {
                    _slots[i].inUse = true;
                    return &_slots[i];
                }
            }
            _cv.wait(lock);
        }
    }
    void release(GpuSlot* s)
    {
        {
            std::lock_guard<std::mutex> lock(_m);
            s->inUse = false;
        }
        _cv.notify_one();
    }
private:
    std::mutex _m;
    std::condition_variable _cv;
    GpuSlot _slots[CPG_NUM_SLOTS];
};

// intentionally leaked: CUDA may already be shut down when static destructors run
SlotPool& slotPool()
{
    static SlotPool* p = new SlotPool;
    return *p;
}

struct SlotGuard
{
    GpuSlot* slot;
    SlotGuard() : slot( slotPool().acquire() ) {}
    ~SlotGuard()
    {
        // leave the slot idle: make sure nothing is pending
        if (slot->stream) {
            cudaStreamSynchronize(slot->stream);
        }
        if (slot->trimRequested) {
            slot->releaseBuffers();
            slot->trimRequested = false;
            cudaGetLastError();
        }
        slotPool().release(slot);
    }
};

////////////////////////////////////////////////////////////////////////////////
// profiling (CORNERPINEX_OFX_PROFILE=1)

bool profilingEnabledImpl()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("CORNERPINEX_OFX_PROFILE");
        enabled = (v && *v && *v != '0') ? 1 : 0;
    }
    return enabled == 1;
}

void profileLog(const std::string& line)
{
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    const char* tmp = std::getenv("TEMP");
    if (!tmp) {
        tmp = std::getenv("TMP");
    }
    std::string path = std::string(tmp ? tmp : "/tmp") + "/CornerPinEx_OFX_profile.log";
    FILE* f = std::fopen(path.c_str(), "a");
    if (f) {
        std::fprintf(f, "%s\n", line.c_str());
        std::fclose(f);
    }
}

struct Profiler
{
    bool on;
    cudaStream_t s;
    std::chrono::steady_clock::time_point t0, last;
    std::ostringstream os;
    Profiler(bool enabled, cudaStream_t stream) : on(enabled), s(stream)
    {
        t0 = last = std::chrono::steady_clock::now();
    }
    double* hostCopyMs = NULL;
    void step(const char* name)
    {
        if (!on) {
            return;
        }
        cudaStreamSynchronize(s);
        auto now = std::chrono::steady_clock::now();
        os << " " << name << "=" << std::chrono::duration<double, std::milli>(now - last).count() << "ms";
        if ( hostCopyMs && (*hostCopyMs > 0.) ) {
            os << "(" << (transferMode() == eTransferRegister ? "page-lock " : "cpu copy ") << *hostCopyMs << "ms)";
            *hostCopyMs = 0.;
        }
        last = now;
    }
    double total() const
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

} // anonymous namespace

bool cudaCompiledIn()
{
    return true;
}

CudaDeviceInfo cudaQueryDevice(int requestedDevice)
{
    CudaDeviceInfo info;
    info.available = false;
    info.device = -1;
    info.totalMem = 0;
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        info.message = errString("init", e);
        cudaGetLastError();
        return info;
    }
    if (n <= 0) {
        info.message = "No CUDA device found";
        return info;
    }
    int dev = (requestedDevice >= 0 && requestedDevice < n) ? requestedDevice : 0;
    cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) {
        info.message = errString("device properties", e);
        cudaGetLastError();
        return info;
    }
    info.available = true;
    info.device = dev;
    info.name = prop.name;
    info.totalMem = prop.totalGlobalMem;
    std::ostringstream os;
    os << "CUDA device " << dev << ": " << prop.name << " (sm_" << prop.major << prop.minor << ", "
       << (prop.totalGlobalMem >> 20) << " MB)";
    if (requestedDevice >= n) {
        os << " - requested device " << requestedDevice << " does not exist";
    }
    info.message = os.str();
    return info;
}

bool cudaRender(const RenderJob& job, int device, std::string* err)
{
    const int NC = job.nComp;
    const int rw = job.rw[2] - job.rw[0];
    const int rh = job.rw[3] - job.rw[1];
    if ( (rw <= 0) || (rh <= 0) ) {
        return true;
    }
    if ( (NC < 1) || (NC > 4) || (job.count() < 1) ) {
        if (err) {
            *err = "unsupported image";
        }
        return false;
    }

    SlotGuard guard;
    GpuSlot& slot = *guard.slot;

#define CPG_CHECK(what, expr) \
    do { cudaError_t ee = (expr); if (ee != cudaSuccess) { if (err) { *err = errString(what, ee); } cudaGetLastError(); return false; } } while (0)
#define CPG_GET(ptr, type, id, bytes) \
    type* ptr = NULL; \
    do { cudaError_t ee = cudaSuccess; ptr = (type*)slot.get(id, (bytes), &ee); \
         if (!ptr) { if (err) { *err = errString("out of memory", ee); } cudaGetLastError(); return false; } } while (0)

    CPG_CHECK( "init", slot.init(device) );
    const cudaStream_t s = slot.stream;
    Profiler prof(profilingEnabledImpl(), s);
    slot.hostCopyMs = 0.;
    prof.hostCopyMs = &slot.hostCopyMs;

    const size_t bpc = bytesPerComp(job.depth);
    const size_t pixBytes = bpc * NC;
    const float maxValue = job.maxValue();

    // 1. source -> float level 0 (+ mip pyramid for the EWA filters), all levels in one buffer
    Pyramid pyr;
    pyr.nLevels = 0;
    pyr.x1 = job.src.x1;
    pyr.y1 = job.src.y1;
    for (int l = 0; l < CPG_MAX_LEVELS; ++l) {
        pyr.level[l].data = NULL;
        pyr.level[l].width = pyr.level[l].height = 0;
        pyr.level[l].stride = 0;
    }
    const bool hasSrc = job.src.data && job.src.width() > 0 && job.src.height() > 0;
    if (hasSrc) {
        const int sw = job.src.width(), sh = job.src.height();
        // level sizes
        int lw[CPG_MAX_LEVELS], lh[CPG_MAX_LEVELS];
        size_t loff[CPG_MAX_LEVELS];
        int nLevels = 1;
        lw[0] = sw; lh[0] = sh; loff[0] = 0;
        size_t total = (size_t)sw * sh * NC;
        if ( isEWA(job.sp.filter) ) {
            while ( (nLevels < CPG_MAX_LEVELS) && ( (lw[nLevels - 1] > 1) || (lh[nLevels - 1] > 1) ) ) {
                lw[nLevels] = (lw[nLevels - 1] + 1) / 2;
                lh[nLevels] = (lh[nLevels - 1] + 1) / 2;
                loff[nLevels] = total;
                total += (size_t)lw[nLevels] * lh[nLevels] * NC;
                ++nLevels;
            }
        }
        CPG_GET(levels, float, eBufLevels, total * sizeof(float));
        if (job.depth == eDepthFloat) {
            CPG_CHECK( "upload", slot.upload(levels, job.src.data, job.src.rowBytes, sw * pixBytes, sh) );
        } else {
            CPG_GET(raw, unsigned char, eBufSrcRaw, (size_t)sw * sh * pixBytes);
            CPG_CHECK( "upload", slot.upload(raw, job.src.data, job.src.rowBytes, sw * pixBytes, sh) );
            CPG_DISPATCH_NC(NC, launchToFloat, job.depth, raw, levels, sw, sh, s);
        }
        prof.step("upload");
        for (int l = 0; l < nLevels; ++l) {
            pyr.level[l].data = levels + loff[l];
            pyr.level[l].width = lw[l];
            pyr.level[l].height = lh[l];
            pyr.level[l].stride = (long long)lw[l] * NC;
            if (l > 0) {
                CPG_DISPATCH_NC(NC, launchDownsample, pyr.level[l - 1], levels + loff[l], lw[l], lh[l], s);
            }
        }
        pyr.nLevels = nLevels;
        if (nLevels > 1) {
            prof.step("mipmaps");
        }
    }

    // 2. transforms
    const int count = job.count();
    std::vector<Mat3> hostH(count);
    for (int t = 0; t < count; ++t) {
        for (int k = 0; k < 9; ++k) {
            hostH[t].m[k] = (float)job.H[t * 9 + k];
        }
    }
    CPG_GET(dH, Mat3, eBufH, count * sizeof(Mat3));
    CPG_CHECK( "upload", cudaMemcpyAsync(dH, hostH.data(), count * sizeof(Mat3), cudaMemcpyHostToDevice, s) );
    float* dAlpha = NULL;
    std::vector<float> hostAlpha;
    if ( !job.alpha.empty() ) {
        hostAlpha.resize( job.alpha.size() );
        for (size_t i = 0; i < job.alpha.size(); ++i) {
            hostAlpha[i] = (float)job.alpha[i];
        }
        CPG_GET(da, float, eBufAlpha, hostAlpha.size() * sizeof(float));
        dAlpha = da;
        CPG_CHECK( "upload", cudaMemcpyAsync(dAlpha, hostAlpha.data(), hostAlpha.size() * sizeof(float), cudaMemcpyHostToDevice, s) );
    }

    // 3. mask
    float* dMask = NULL;
    const bool hasMask = job.masked && job.doMask && job.mask.data && job.mask.width() > 0 && job.mask.height() > 0;
    if (hasMask) {
        const int mw = job.mask.width(), mh = job.mask.height();
        CPG_GET(dm, float, eBufMask, (size_t)mw * mh * sizeof(float));
        dMask = dm;
        CPG_GET(raw, unsigned char, eBufMaskRaw, (size_t)mw * mh * bpc);
        CPG_CHECK( "upload", slot.upload(raw, job.mask.data, job.mask.rowBytes, mw * bpc, mh) );
        dim3 b(32, 8);
        const float scale = 1.f / maxValue;
        if (job.depth == eDepthUByte) {
            kMaskToFloat<unsigned char><<<grid2d(mw, mh, b), b, 0, s>>>( (const unsigned char*)raw, dMask, mw, mh, scale, job.maskInvert );
        } else if (job.depth == eDepthUShort) {
            kMaskToFloat<unsigned short><<<grid2d(mw, mh, b), b, 0, s>>>( (const unsigned short*)raw, dMask, mw, mh, scale, job.maskInvert );
        } else {
            kMaskToFloat<float><<<grid2d(mw, mh, b), b, 0, s>>>( (const float*)raw, dMask, mw, mh, scale, job.maskInvert );
        }
        prof.step("mask");
    }

    // 4. render
    KernelArgs a;
    a.pyr = pyr;
    a.sp = job.sp;
    a.H = dH;
    a.alpha = dAlpha;
    a.count = count;
    a.motionblur = (float)job.motionblur;
    a.maxValue = maxValue;
    a.rx1 = job.rw[0];
    a.ry1 = job.rw[1];
    a.rw = rw;
    a.rh = rh;
    a.mask = dMask;
    a.mx1 = job.mask.x1;
    a.my1 = job.mask.y1;
    a.mw = hasMask ? job.mask.width() : 0;
    a.mh = hasMask ? job.mask.height() : 0;
    a.masked = job.masked;
    a.doMask = job.masked && job.doMask;
    a.maskInvert = job.maskInvert;
    a.mix = (float)job.mix;
    a.hasSrc = hasSrc;

    CPG_GET(dOutF, float, eBufOutF, (size_t)rw * rh * NC * sizeof(float));
    CPG_DISPATCH_NC(NC, launchRender, a, dOutF, s);
    CPG_CHECK( "render kernel", cudaGetLastError() );
    flushStream(s);
    prof.step("kernel");

    // 5. convert and download into the render window of the output image
    unsigned char* dstBase = (unsigned char*)job.dstData
                             + (long long)(job.rw[1] - job.dstY1) * job.dstRowBytes
                             + (long long)(job.rw[0] - job.dstX1) * (long long)pixBytes;
    if (job.depth == eDepthFloat) {
        CPG_CHECK( "download", slot.download(dstBase, job.dstRowBytes, dOutF, rw * pixBytes, rh) );
    } else {
        CPG_GET(dOut, unsigned char, eBufOutRaw, (size_t)rw * rh * pixBytes);
        CPG_DISPATCH_NC(NC, launchFromFloat, job.depth, dOutF, dOut, rw, rh, maxValue, s);
        CPG_CHECK( "download", slot.download(dstBase, job.dstRowBytes, dOut, rw * pixBytes, rh) );
    }
    CPG_CHECK( "synchronize", cudaStreamSynchronize(s) );
    prof.step("download");

    if (prof.on) {
        std::ostringstream os;
        os << "GPU render " << rw << "x" << rh << " nc=" << NC << " depth=" << (int)job.depth
           << " filter=" << job.sp.filter << " transforms=" << count
           << " src=" << job.src.width() << "x" << job.src.height() << " transfer=" << transferModeName( transferMode() ) << " copyThreads=" << copyThreadCount()
           << " |" << prof.os.str() << " total=" << prof.total() << "ms";
        profileLog( os.str() );
    }

#undef CPG_CHECK
#undef CPG_GET
    return true;
} // cudaRender

bool profilingEnabled()
{
    return profilingEnabledImpl();
}

void profileCpuRender(const char* what, int w, int h, double ms)
{
    if ( !profilingEnabledImpl() ) {
        return;
    }
    std::ostringstream os;
    os << "CPU render (" << what << ") " << w << "x" << h << " total=" << ms << "ms";
    profileLog( os.str() );
}
} // namespace cpg
