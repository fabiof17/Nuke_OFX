// DespillITCPU.cpp
// Multithreaded CPU back-end of despill_IT_OFX (same stages as the CUDA back-end).
#include "DespillITCore.h"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

namespace dspit {
namespace {

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

// Push-pull in place on an RGBW image (premultiplied colour, weight).
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

// Distance (px) of every pixel to the nearest screen pixel (mask > 0.5).
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

// 3 box passes along the rows then the columns (nch channels, pstride floats per pixel).
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

} // namespace

void processCPU(const Params& p, const Inputs& in, float* out)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0 || !in.src) return;
    const size_t N = (size_t)w * h;

    // 1. key with the global screen colour -> screen seeds -> local screen (push-pull)
    std::vector<float> pyr;
    if (needsLocalScreen(p)) {
        pyr.resize(4 * N);
        parallelFor(h, [&](int y0, int y1) {
            for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i)
                keyPass1(p, in.src + 4 * i, in.key ? in.key + i : nullptr, &pyr[4 * i]);
        });
        pushPull(pyr, w, h);
    }

    // 2. local screen, final key, screen mask, core seeds
    std::vector<float> screen(4 * N), alpha(N), screenMask(N), core(4 * N);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
            float* S = &screen[4 * i];
            localScreen(p, pyr.empty() ? nullptr : &pyr[4 * i], in.clean ? in.clean + 3 * i : nullptr, S);
            S[3] = 0.f;
            const float pr = protection(p, in.src + 4 * i, in.protect ? in.protect + i : nullptr);
            alpha[i] = keyPass2(p, in.src + 4 * i, in.key ? in.key + i : nullptr, S, pr, &core[4 * i], &screenMask[i]);
        }
    });

    // 3. distance to the screen
    std::vector<float> dist;
    if (p.spatial) distanceToScreen(screenMask, w, h, dist);

    // 4. opaque core colour pushed into the edges
    if (needsCore(p)) pushPull(core, w, h);

    // 5. blurred background and light-wrap edge, on an image reduced by f
    int f, br;
    bgReducePlan(p.bgBlur, &f, &br);
    const int ws = (w + f - 1) / f, hs = (h + f - 1) / f;
    std::vector<float> bgSmall, alphaSmall;
    const bool useBG = needsBGBlur(p) && in.bg, useWrap = needsWrap(p) && in.bg;
    if (useBG) {
        bgSmall.resize((size_t)3 * ws * hs);
        parallelFor(hs, [&](int y0, int y1) {
            for (int Y = y0; Y < y1; ++Y)
                for (int X = 0; X < ws; ++X) downsamplePixel(in.bg, w, h, 3, f, X, Y, &bgSmall[3 * ((size_t)Y * ws + X)]);
        });
        if (br > 0) blur(bgSmall, ws, hs, 3, 3, br);
    }
    if (useWrap) {
        alphaSmall.resize((size_t)ws * hs);
        parallelFor(hs, [&](int y0, int y1) {
            for (int Y = y0; Y < y1; ++Y)
                for (int X = 0; X < ws; ++X) downsamplePixel(alpha.data(), w, h, 1, f, X, Y, &alphaSmall[(size_t)Y * ws + X]);
        });
        if (br > 0) blur(alphaSmall, ws, hs, 1, 1, br);
    }

    // 6. final colour
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                const size_t i = (size_t)y * w + x;
                float bg3[3], a1 = 1.f;
                if (useBG) sampleReduced(bgSmall.data(), ws, hs, 3, f, x, y, bg3);
                if (useWrap) sampleReduced(alphaSmall.data(), ws, hs, 1, f, x, y, &a1);
                finalPixel(p, in.src + 4 * i, &screen[4 * i], alpha[i], dist.empty() ? 0.f : dist[i],
                           needsCore(p) ? &core[4 * i] : nullptr, useBG ? bg3 : nullptr, 1.f - a1,
                           in.protect ? in.protect + i : nullptr, in.mask ? in.mask[i] : 1.f, out + 4 * i);
            }
    });
}

} // namespace dspit
