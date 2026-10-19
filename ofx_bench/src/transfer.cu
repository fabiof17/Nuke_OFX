// Host <-> GPU transfer speed, pageable vs pinned memory (benchmark only).
#include <cuda_runtime.h>
#include "push_it/PinnedStaging.cuh"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

std::string transferTest(size_t bytes, int runs)
{
    void* dev = nullptr;
    if (cudaMalloc(&dev, bytes) != cudaSuccess) return "      transfer test: cudaMalloc failed\n";
    std::vector<char> pageable(bytes, 1);
    void* pinned = nullptr;
    const bool havePinned = cudaMallocHost(&pinned, bytes) == cudaSuccess;
    if (havePinned) std::memset(pinned, 1, bytes);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    auto time = [&](void* dst, const void* src, cudaMemcpyKind kind) {
        float best = 1e30f;
        for (int i = 0; i < runs; ++i) {
            cudaEventRecord(a);
            cudaMemcpy(dst, src, bytes, kind);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, a, b);
            best = ms < best ? ms : best;
        }
        return best;
    };
    const double gb = (double)bytes / 1e9;
    char buf[512];
    const float up = time(dev, pageable.data(), cudaMemcpyHostToDevice);
    const float down = time(pageable.data(), dev, cudaMemcpyDeviceToHost);
    std::string r;
    std::snprintf(buf, sizeof buf, "      pageable          : up %6.2f ms (%5.1f GB/s)   down %6.2f ms (%5.1f GB/s)\n",
                  up, gb / (up / 1e3), down, gb / (down / 1e3));
    r += buf;
    if (havePinned) {
        const float pu = time(dev, pinned, cudaMemcpyHostToDevice);
        const float pd = time(pinned, dev, cudaMemcpyDeviceToHost);
        std::snprintf(buf, sizeof buf, "      pinned            : up %6.2f ms (%5.1f GB/s)   down %6.2f ms (%5.1f GB/s)\n",
                      pu, gb / (pu / 1e3), pd, gb / (pd / 1e3));
        r += buf;
        cudaFreeHost(pinned);
    }
    // the path used by the plug-ins: pinned staging buffers, CPU copy split over N threads
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    for (int t : { 1, 2, 4, 6 }) {
        PinnedStaging stg(t);
        auto wall = [&](bool up) {
            double best = 1e30;
            for (int i = 0; i < runs + 1; ++i) {           // first run allocates the buffers
                const auto t0 = std::chrono::steady_clock::now();
                if (up) { stg.upload(dev, pageable.data(), bytes, st); cudaStreamSynchronize(st); }
                else    stg.download(pageable.data(), dev, bytes, st);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (i > 0 && ms < best) best = ms;
            }
            return best;
        };
        const double su = wall(true), sd = wall(false);
        std::snprintf(buf, sizeof buf, "      staged, %d thread%s : up %6.2f ms (%5.1f GB/s)   down %6.2f ms (%5.1f GB/s)\n",
                      t, t > 1 ? "s" : " ", su, gb / (su / 1e3), sd, gb / (sd / 1e3));
        r += buf;
        stg.release();
    }
    cudaStreamDestroy(st);
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dev);
    return r;
}
