// PinnedStaging.cuh
// Fast host <-> GPU transfers for images that live in ordinary (pageable) host memory,
// which is what Nuke hands to OFX plug-ins.
//
// A pageable copy runs at about half the bus speed because the driver first copies the
// data into its own small page-locked buffer. Here we do that ourselves with two larger
// page-locked (pinned) buffers used in turn: while the GPU pulls chunk i from one buffer,
// the CPU fills chunk i+1 into the other, so the CPU work and the DMA overlap.
//
// The row variants let the caller write its rows straight into the pinned buffer (and read
// the result straight out of it): the plug-ins convert Nuke's images there directly, with
// no intermediate copy of the whole image. The CPU side is split over a few threads.
// If pinned memory cannot be allocated, a plain temporary buffer is used instead.
#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// Splits [0, n) into `threads` contiguous parts run in parallel (the caller is one of the
// threads). Workers are persistent.
class WorkerPool {
public:
    explicit WorkerPool(int threads) : threads_(threads < 1 ? 1 : threads) {}
    ~WorkerPool() { stop(); }

    // f(b, e) for parts of [0, n); parts are multiples of `grain` (except the last one).
    void run(size_t n, size_t grain, const std::function<void(size_t, size_t)>& f)
    {
        if (n == 0) return;
        if (threads_ == 1 || n <= grain) { f(0, n); return; }
        start();
        {
            std::lock_guard<std::mutex> lk(m_);
            f_ = &f; n_ = n; grain_ = grain;
            pending_ = (int)workers_.size();
            ++gen_;
        }
        cv_.notify_all();
        part(0);
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return pending_ == 0; });
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        quit_ = false;
    }

private:
    void start()
    {
        if (!workers_.empty()) return;
        for (int i = 1; i < threads_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }
    void part(int i)
    {
        const size_t units = (n_ + grain_ - 1) / grain_;
        const size_t per = (units + threads_ - 1) / threads_ * grain_;
        const size_t b = per * (size_t)i;
        if (b >= n_) return;
        const size_t e = b + per < n_ ? b + per : n_;
        (*f_)(b, e);
    }
    void loop(int i)
    {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
                if (quit_) return;
                seen = gen_;
            }
            part(i);
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) done_.notify_one();
        }
    }

    const int threads_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    uint64_t gen_ = 0;
    int pending_ = 0;
    bool quit_ = false;
    const std::function<void(size_t, size_t)>* f_ = nullptr;
    size_t n_ = 0, grain_ = 1;
};

class PinnedStaging {
public:
    // fill(y0, y1, dst): writes rows [y0, y1) contiguously at dst (rowBytes each).
    using RowFill = std::function<void(int, int, void*)>;
    // drain(y0, y1, src): reads rows [y0, y1) stored contiguously at src.
    using RowDrain = std::function<void(int, int, const void*)>;

    // threads: CPU threads used on the host side (0 = automatic); chunk: bytes per buffer.
    explicit PinnedStaging(int threads = 0, size_t chunk = (size_t)8 << 20)
        : pool_(threads > 0 ? threads : autoThreads()), chunk_(chunk) {}

    void release()
    {
        freeBuffers();
        pool_.stop();
    }

    // host rows -> device (rows * rowBytes bytes at dDst).
    cudaError_t uploadRows(void* dDst, size_t rowBytes, int rows, cudaStream_t st, const RowFill& fill)
    {
        if (rows <= 0 || rowBytes == 0) return cudaSuccess;
        if (!ready(rowBytes)) {                           // no pinned memory: temporary buffer
            std::vector<char> tmp(rowBytes * (size_t)rows);
            fillParallel(fill, 0, rows, rowBytes, tmp.data());
            cudaError_t e = cudaMemcpyAsync(dDst, tmp.data(), tmp.size(), cudaMemcpyHostToDevice, st);
            return e != cudaSuccess ? e : cudaStreamSynchronize(st);
        }
        const int per = (int)(size_ / rowBytes);
        char* d = static_cast<char*>(dDst);
        int k = 0;
        for (int y = 0; y < rows; y += per, k ^= 1) {
            const int n = rows - y < per ? rows - y : per;
            cudaError_t e = cudaEventSynchronize(ev_[k]);   // buffer k no longer read by the GPU
            if (e != cudaSuccess) return e;
            fillParallel(fill, y, y + n, rowBytes, buf_[k]);
            if ((e = cudaMemcpyAsync(d + (size_t)y * rowBytes, buf_[k], (size_t)n * rowBytes,
                                     cudaMemcpyHostToDevice, st)) != cudaSuccess) return e;
            if ((e = cudaEventRecord(ev_[k], st)) != cudaSuccess) return e;
        }
        return cudaSuccess;
    }

    // device (rows * rowBytes bytes at dSrc) -> host rows. Waits for the stream.
    cudaError_t downloadRows(const void* dSrc, size_t rowBytes, int rows, cudaStream_t st, const RowDrain& drain)
    {
        if (rows <= 0 || rowBytes == 0) return cudaStreamSynchronize(st);
        if (!ready(rowBytes)) {
            std::vector<char> tmp(rowBytes * (size_t)rows);
            cudaError_t e = cudaMemcpyAsync(tmp.data(), dSrc, tmp.size(), cudaMemcpyDeviceToHost, st);
            if (e == cudaSuccess) e = cudaStreamSynchronize(st);
            if (e == cudaSuccess) drainParallel(drain, 0, rows, rowBytes, tmp.data());
            return e;
        }
        const int per = (int)(size_ / rowBytes);
        const char* s = static_cast<const char*>(dSrc);
        const int chunks = (rows + per - 1) / per;
        auto issue = [&](int i) -> cudaError_t {
            const int y = i * per, n = rows - y < per ? rows - y : per;
            cudaError_t e = cudaMemcpyAsync(buf_[i & 1], s + (size_t)y * rowBytes, (size_t)n * rowBytes,
                                            cudaMemcpyDeviceToHost, st);
            return e != cudaSuccess ? e : cudaEventRecord(ev_[i & 1], st);
        };
        cudaError_t e = issue(0);
        for (int i = 0; i < chunks && e == cudaSuccess; ++i) {
            if (i + 1 < chunks && (e = issue(i + 1)) != cudaSuccess) break;   // GPU copies the next chunk...
            if ((e = cudaEventSynchronize(ev_[i & 1])) != cudaSuccess) break;
            const int y = i * per, n = rows - y < per ? rows - y : per;
            drainParallel(drain, y, y + n, rowBytes, buf_[i & 1]);              // ...while the CPU reads this one
        }
        return e;
    }

    // Plain contiguous copies (rows of 64 KB).
    cudaError_t upload(void* dDst, const void* hSrc, size_t bytes, cudaStream_t st)
    {
        const char* s = static_cast<const char*>(hSrc);
        const size_t rb = kRawRow;
        const int full = (int)(bytes / rb);
        cudaError_t e = uploadRows(dDst, rb, full, st, [&](int y0, int y1, void* dst) {
            std::memcpy(dst, s + (size_t)y0 * rb, (size_t)(y1 - y0) * rb);
        });
        const size_t rest = bytes - (size_t)full * rb;
        if (e == cudaSuccess && rest)
            e = uploadRows(static_cast<char*>(dDst) + (size_t)full * rb, rest, 1, st, [&](int, int, void* dst) {
                std::memcpy(dst, s + (size_t)full * rb, rest);
            });
        return e;
    }
    cudaError_t download(void* hDst, const void* dSrc, size_t bytes, cudaStream_t st)
    {
        char* d = static_cast<char*>(hDst);
        const size_t rb = kRawRow;
        const int full = (int)(bytes / rb);
        cudaError_t e = downloadRows(dSrc, rb, full, st, [&](int y0, int y1, const void* src) {
            std::memcpy(d + (size_t)y0 * rb, src, (size_t)(y1 - y0) * rb);
        });
        const size_t rest = bytes - (size_t)full * rb;
        if (e == cudaSuccess && rest)
            e = downloadRows(static_cast<const char*>(dSrc) + (size_t)full * rb, rest, 1, st,
                             [&](int, int, const void* src) { std::memcpy(d + (size_t)full * rb, src, rest); });
        return e;
    }

private:
    static constexpr size_t kRawRow = (size_t)64 << 10;

    static int autoThreads()
    {
        const unsigned hw = std::thread::hardware_concurrency();
        const int t = hw ? (int)hw / 2 : 2;           // about one per physical core...
        return t < 1 ? 1 : (t > 4 ? 4 : t);           // ...memory bandwidth is saturated beyond 4
    }

    void fillParallel(const RowFill& fill, int y0, int y1, size_t rowBytes, void* dst)
    {
        char* base = static_cast<char*>(dst);
        const size_t grain = rowBytes >= ((size_t)1 << 20) ? 1 : ((size_t)1 << 20) / rowBytes;   // ~1 MB per part
        pool_.run((size_t)(y1 - y0), grain, [&](size_t b, size_t e) {
            fill(y0 + (int)b, y0 + (int)e, base + b * rowBytes);
        });
    }
    void drainParallel(const RowDrain& drain, int y0, int y1, size_t rowBytes, const void* src)
    {
        const char* base = static_cast<const char*>(src);
        const size_t grain = rowBytes >= ((size_t)1 << 20) ? 1 : ((size_t)1 << 20) / rowBytes;
        pool_.run((size_t)(y1 - y0), grain, [&](size_t b, size_t e) {
            drain(y0 + (int)b, y0 + (int)e, base + b * rowBytes);
        });
    }

    void freeBuffers()
    {
        for (int k = 0; k < 2; ++k) {
            if (ev_[k]) cudaEventSynchronize(ev_[k]);     // the GPU may still be reading it
            if (buf_[k]) cudaFreeHost(buf_[k]);
            if (ev_[k]) cudaEventDestroy(ev_[k]);
            buf_[k] = nullptr; ev_[k] = nullptr;
        }
        size_ = 0;
        failed_ = false;
    }

    // Pinned buffers holding at least one row.
    bool ready(size_t rowBytes)
    {
        if (buf_[0] && size_ >= rowBytes) return true;
        if (failed_) return false;
        if (buf_[0]) freeBuffers();                        // a row bigger than the buffers
        const size_t want = rowBytes > chunk_ ? rowBytes : chunk_;
        bool ok = true;
        for (int k = 0; k < 2 && ok; ++k)
            ok = cudaMallocHost(&buf_[k], want) == cudaSuccess &&
                 cudaEventCreateWithFlags(&ev_[k], cudaEventDisableTiming) == cudaSuccess;
        if (!ok) { cudaGetLastError(); freeBuffers(); failed_ = true; return false; }
        size_ = want;
        return true;
    }

    WorkerPool pool_;
    size_t chunk_;
    size_t size_ = 0;
    void* buf_[2] = { nullptr, nullptr };
    cudaEvent_t ev_[2] = { nullptr, nullptr };
    bool failed_ = false;
};

} // namespace
