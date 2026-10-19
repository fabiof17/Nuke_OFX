// DespillCPU.cpp
// Multithreaded CPU back-end of L_Despill_OFX.
#include "DespillCore.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace ldespill {
namespace {

// f(b, e) on contiguous ranges of [0, n) across hardware threads.
template <class F>
void parallelFor(int n, F&& f, size_t work = 1u << 20)
{
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 2 || nt == 1 || work < 65536) { f(0, n); return; }
    nt = std::min<unsigned>(nt, (unsigned)n);
    const int chunk = (n + (int)nt - 1) / (int)nt;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t) {
        const int b = (int)t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    for (auto& th : pool) th.join();
}

// Median of the (2r+1)^2 window, edges repeated.
void median(const float* in, float* out, int w, int h, int r)
{
    parallelFor(h, [&](int y0, int y1) {
        std::vector<float> v((size_t)(2 * r + 1) * (2 * r + 1));
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                size_t n = 0;
                for (int dy = -r; dy <= r; ++dy) {
                    const float* row = in + (size_t)ldClampi(y + dy, 0, h - 1) * w;
                    for (int dx = -r; dx <= r; ++dx) v[n++] = row[ldClampi(x + dx, 0, w - 1)];
                }
                std::nth_element(v.begin(), v.begin() + n / 2, v.begin() + n);
                out[(size_t)y * w + x] = v[n / 2];
            }
    }, (size_t)w * h * (2 * r + 1) * (2 * r + 1));
}

// Min (erode) or max (dilate) over a square of radius r: horizontal then vertical pass.
void minMax(float* img, float* tmp, int w, int h, int r, bool dilate)
{
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const float* row = img + (size_t)y * w;
            for (int x = 0; x < w; ++x) {
                float m = row[ldClampi(x - r, 0, w - 1)];
                for (int d = -r + 1; d <= r; ++d) {
                    const float v = row[ldClampi(x + d, 0, w - 1)];
                    m = dilate ? (v > m ? v : m) : (v < m ? v : m);
                }
                tmp[(size_t)y * w + x] = m;
            }
        }
    }, (size_t)w * h * r);
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float m = tmp[(size_t)ldClampi(y - r, 0, h - 1) * w + x];
                for (int d = -r + 1; d <= r; ++d) {
                    const float v = tmp[(size_t)ldClampi(y + d, 0, h - 1) * w + x];
                    m = dilate ? (v > m ? v : m) : (v < m ? v : m);
                }
                img[(size_t)y * w + x] = m;
            }
    }, (size_t)w * h * r);
}

// 3 box passes along the rows, then 3 along the columns (as the CUDA back-end).
void blur(float* img, float* tmp, int w, int h, int r)
{
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            float* a = img + (size_t)y * w;
            float* b = tmp + (size_t)y * w;
            slideBox(a, b, w, 1, r); slideBox(b, a, w, 1, r); slideBox(a, b, w, 1, r);
            std::copy(b, b + w, a);
        }
    }, (size_t)w * h);
    parallelFor(w, [&](int x0, int x1) {
        for (int x = x0; x < x1; ++x) {
            slideBox(img + x, tmp + x, h, w, r); slideBox(tmp + x, img + x, h, w, r); slideBox(img + x, tmp + x, h, w, r);
            for (int y = 0; y < h; ++y) img[(size_t)y * w + x] = tmp[(size_t)y * w + x];
        }
    }, (size_t)w * h);
}

} // namespace

void processCPU(const Params& p, const float* src, const float* bg, const float* mask, float* out)
{
    const int w = p.bufW, h = p.bufH;
    if (w <= 0 || h <= 0 || p.winW <= 0 || p.winH <= 0) return;
    const size_t N = (size_t)w * h;

    // 1. raw spill matte over the whole buffer
    std::vector<float> matte(N), tmp(N);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) matte[i] = matteRawPixel(p, src + 4 * i);
    }, N);

    // 2. spatial filters
    if (p.medianRadius > 0) { median(matte.data(), tmp.data(), w, h, p.medianRadius); matte.swap(tmp); }
    if (p.cleanRadius != 0) {
        const int r = std::abs(p.cleanRadius);
        const bool close = p.cleanRadius > 0;       // close: dilate then erode / open: erode then dilate
        minMax(matte.data(), tmp.data(), w, h, r, close);
        minMax(matte.data(), tmp.data(), w, h, r, !close);
    }
    const int br = boxRadiusForBlur(p.matteBlur);
    if (br > 0) blur(matte.data(), tmp.data(), w, h, br);

    // 3. final colour over the output window
    parallelFor(p.winH, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < p.winW; ++x) {
                const size_t bi = (size_t)(y + p.winY) * w + (x + p.winX);
                const size_t wi = (size_t)y * p.winW + x;
                finalPixel(p, src + 4 * bi, bg ? bg + 3 * wi : nullptr, matte[bi], mask ? mask[wi] : 1.f, out + 4 * wi);
            }
    }, (size_t)p.winW * p.winH);
}

} // namespace ldespill
