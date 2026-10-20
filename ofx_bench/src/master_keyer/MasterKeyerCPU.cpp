// MasterKeyerCPU.cpp
// Multithreaded CPU back-end of MasterKeyer_OFX (same stages as the CUDA back-end).
#include "MasterKeyerCore.h"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

namespace mk {
namespace {

using namespace mkshared;

template <class F>
void parallelFor(int n, F&& f)
{
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 2 || nt == 1) { f(0, n); return; }
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

void pushPull(std::vector<float>& img, int w, int h)
{
    std::vector<std::vector<float>> levels;
    std::vector<int> lw(1, w), lh(1, h);
    levels.push_back(std::move(img));
    while (lw.back() > 1 || lh.back() > 1) {
        const int cw = (lw.back() + 1) / 2, ch = (lh.back() + 1) / 2;
        std::vector<float> c((size_t)4 * cw * ch);
        const std::vector<float>& f = levels.back();
        const int fw = lw.back(), fh = lh.back();
        parallelFor(ch, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < cw; ++x) pullPixel(f.data(), fw, fh, x, y, &c[4 * ((size_t)y * cw + x)]);
        });
        levels.push_back(std::move(c));
        lw.push_back(cw); lh.push_back(ch);
    }
    for (int l = (int)levels.size() - 2; l >= 0; --l) {
        std::vector<float>& f = levels[l];
        const std::vector<float>& c = levels[l + 1];
        const int fw = lw[l], fh = lh[l], cw = lw[l + 1], ch = lh[l + 1];
        parallelFor(fh, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < fw; ++x) pushPixel(&f[4 * ((size_t)y * fw + x)], c.data(), cw, ch, x, y);
        });
    }
    img = std::move(levels[0]);
}

void distanceToScreen(const std::vector<float>& mask, int w, int h, std::vector<float>& dist)
{
    const size_t N = (size_t)w * h;
    std::vector<int> nearestY(N);
    parallelFor(w, [&](int x0, int x1) { for (int x = x0; x < x1; ++x) edtColumn(mask.data(), w, h, x, nearestY.data()); });
    std::vector<Seed> seeds(N);
    parallelFor(h, [&](int y0, int y1) {
        std::vector<int> v(w), zd(w);
        std::vector<long long> zn(w);
        for (int y = y0; y < y1; ++y)
            edtRow(nearestY.data() + (size_t)y * w, w, y, 1, v.data(), zn.data(), zd.data(), seeds.data() + (size_t)y * w);
    });
    dist.resize(N);
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) dist[(size_t)y * w + x] = seedDistance(seeds[(size_t)y * w + x], x, y);
    });
}

void blur(std::vector<float>& img, int w, int h, int pstride, int nch, int r)
{
    std::vector<float> tmp(img.size());
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            float* a = &img[(size_t)y * w * pstride];
            float* b = &tmp[(size_t)y * w * pstride];
            slideBox(a, b, w, pstride, nch, r); slideBox(b, a, w, pstride, nch, r); slideBox(a, b, w, pstride, nch, r);
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < nch; ++c) a[(size_t)x * pstride + c] = b[(size_t)x * pstride + c];
        }
    });
    const size_t stride = (size_t)w * pstride;
    parallelFor(w, [&](int x0, int x1) {
        for (int x = x0; x < x1; ++x) {
            float* a = &img[(size_t)x * pstride];
            float* b = &tmp[(size_t)x * pstride];
            slideBox(a, b, h, stride, nch, r); slideBox(b, a, h, stride, nch, r); slideBox(a, b, h, stride, nch, r);
            for (int y = 0; y < h; ++y)
                for (int c = 0; c < nch; ++c) a[(size_t)y * stride + c] = b[(size_t)y * stride + c];
        }
    });
}

// Min (erode) or max (dilate) over a square of radius r, rows then columns, edges repeated.
void minMax(std::vector<float>& img, int w, int h, int r, bool dilate)
{
    std::vector<float> tmp(img.size());
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float m = img[(size_t)y * w + diClampi(x - r, 0, w - 1)];
                for (int d = -r + 1; d <= r; ++d) {
                    const float v = img[(size_t)y * w + diClampi(x + d, 0, w - 1)];
                    m = dilate ? (v > m ? v : m) : (v < m ? v : m);
                }
                tmp[(size_t)y * w + x] = m;
            }
    });
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                float m = tmp[(size_t)diClampi(y - r, 0, h - 1) * w + x];
                for (int d = -r + 1; d <= r; ++d) {
                    const float v = tmp[(size_t)diClampi(y + d, 0, h - 1) * w + x];
                    m = dilate ? (v > m ? v : m) : (v < m ? v : m);
                }
                img[(size_t)y * w + x] = m;
            }
    });
}

} // namespace

void processCPU(const Params& p, const Inputs& in, float* out)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0 || !in.src) return;
    const size_t N = (size_t)w * h;

    // 1. colour used for the matte: Key In (or the plate), degrained or not
    std::vector<float> mc;
    const float* matteColour = in.keyIn ? in.keyIn : in.src;
    size_t ms = in.keyIn ? 3 : 4;                      // floats per pixel of matteColour
    if (p.degrain) {
        std::vector<float> ycc(4 * N);
        const float* base = matteColour;
        const size_t bs = ms;
        parallelFor(h, [&](int y0, int y1) {
            for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) toYCC(base + bs * i, &ycc[4 * i]);
        });
        std::vector<float> yb = ycc;
        const int r = boxRadiusForBlur(grainBlurSize(p));
        if (r > 0) blur(yb, w, h, 4, degrainChannels(p), r);
        mc.resize(4 * N);
        parallelFor(h, [&](int y0, int y1) {
            for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
                fromYCC(&ycc[4 * i], &yb[4 * i], p.grainEdges, &mc[4 * i]);
                mc[4 * i + 3] = 0.f;
            }
        });
        matteColour = mc.data();
        ms = 4;
    }

    // 2. local screen (push-pull of the screen areas)
    std::vector<float> pushed;
    if (p.screenSource == kScreenLocal) {
        pushed.resize(4 * N);
        parallelFor(h, [&](int y0, int y1) {
            for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) screenSeed(p, matteColour + ms * i, &pushed[4 * i]);
        });
        pushPull(pushed, w, h);
    }

    // 3. matte
    std::vector<float> screen(4 * N), alpha(N);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
            float* S = &screen[4 * i];
            screenAt(p, pushed.empty() ? nullptr : &pushed[4 * i], S);
            S[3] = 0.f;
            alpha[i] = mattePixel(p, matteColour + ms * i, S);
        }
    });
    if (p.shrink != 0) minMax(alpha, w, h, std::abs(p.shrink), p.shrink > 0);
    const int mr = boxRadiusForBlur(p.matteBlur);
    if (mr > 0) blur(alpha, w, h, 1, 1, mr);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i)
            alpha[i] = garbageHoldout(p, alpha[i], in.garbage ? in.garbage + i : nullptr, in.holdout ? in.holdout + i : nullptr);
    });

    // 4. distance to the screen
    std::vector<float> dist;
    if (needsDistance(p)) {
        std::vector<float> mask(N);
        for (size_t i = 0; i < N; ++i) mask[i] = alpha[i] <= 0.02f ? 1.f : 0.f;
        distanceToScreen(mask, w, h, dist);
    }

    // 5. blurred background (reduced)
    int f = 1, br = 0;
    bgReducePlan(p.bgBlur, &f, &br);
    const int ws = (w + f - 1) / f, hs = (h + f - 1) / f;
    std::vector<float> bgSmall;
    const bool useBG = needsBGBlur(p) && in.bg;
    if (useBG) {
        bgSmall.resize((size_t)3 * ws * hs);
        parallelFor(hs, [&](int y0, int y1) {
            for (int Y = y0; Y < y1; ++Y)
                for (int X = 0; X < ws; ++X) downsamplePixel(in.bg, w, h, 3, f, X, Y, &bgSmall[3 * ((size_t)Y * ws + X)]);
        });
        if (br > 0) blur(bgSmall, ws, hs, 3, 3, br);
    }

    // 6. despill, blend, output
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                float b3[3];
                if (useBG) sampleReduced(bgSmall.data(), ws, hs, 3, f, x, y, b3);
                finalPixel(p, in.src + 4 * i, &screen[4 * i], alpha[i], dist.empty() ? 1e30f : dist[i],
                           in.bg ? in.bg + 3 * i : nullptr, useBG ? b3 : nullptr, out + 4 * i);
            }
    });
}

} // namespace mk
