// push_IT_OFXCPU.cpp
// Multithreaded CPU back-end. Same pipeline as push_IT_OFXCUDA.cu.
#include "push_IT_OFXCore.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace pushit {
namespace {

// Runs f(begin, end) over [0, n) split across hardware threads.
template <class F>
void parallelFor(int n, F&& f)
{
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 32 || nt == 1) { f(0, n); return; }
    nt = std::min<unsigned>(nt, (unsigned)((n + 7) / 8));
    const int chunk = (n + (int)nt - 1) / (int)nt;
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
        const int b = (int)t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    for (auto& th : pool) th.join();
}

// Three box passes per direction (Gaussian approximation). The three horizontal passes run
// on one row at a time and the three vertical ones on blocks of columns, in local buffers
// that stay in cache (instead of six sweeps over the whole image).
void gaussianish(std::vector<float>& img, int w, int h, int pstride, int nch, float size)
{
    const int r = boxRadiusForBlur(size);
    if (r <= 0) return;
    float* data = img.data();
    parallelFor(h, [&](int y0, int y1) {
        std::vector<float> a((size_t)w * pstride), b((size_t)w * pstride);
        for (int y = y0; y < y1; ++y) {
            float* row = data + (size_t)y * w * pstride;
            std::copy(row, row + (size_t)w * pstride, a.begin());
            float* in = a.data();
            float* out = b.data();
            for (int pass = 0; pass < 3; ++pass) {
                slideBox(in, out, w, pstride, nch, r);
                std::swap(in, out);
            }
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < nch; ++c) row[(size_t)x * pstride + c] = in[(size_t)x * pstride + c];
        }
    });
    const int block = 32;
    const int nBlocks = (w + block - 1) / block;
    parallelFor(nBlocks, [&](int b0, int b1) {
        std::vector<float> a((size_t)h * block * pstride), b((size_t)h * block * pstride);
        std::vector<double> sums((size_t)block * pstride);
        for (int bi = b0; bi < b1; ++bi) {
            const int x0 = bi * block, bw = std::min(w, x0 + block) - x0;
            const size_t lineStride = (size_t)bw * pstride;
            for (int y = 0; y < h; ++y)
                std::copy(data + ((size_t)y * w + x0) * pstride, data + ((size_t)y * w + x0 + bw) * pstride,
                          a.begin() + (size_t)y * lineStride);
            // the whole block slides down at once: one double sum per (column, channel),
            // same arithmetic and order as slideBox() on each column
            const int m = bw * pstride;
            float* in = a.data();
            float* out = b.data();
            const double norm = 1.0 / (double)(2 * r + 1);
            for (int pass = 0; pass < 3; ++pass) {
                std::fill(sums.begin(), sums.begin() + m, 0.0);
                for (int i = -r; i <= r; ++i) {
                    const float* q = in + (size_t)psClampi(i, 0, h - 1) * lineStride;
                    for (int k = 0; k < m; ++k) sums[k] += q[k];
                }
                for (int y = 0; y < h; ++y) {
                    float* o = out + (size_t)y * lineStride;
                    for (int k = 0; k < m; ++k) o[k] = (float)(sums[k] * norm);
                    const float* qa = in + (size_t)psClampi(y + r + 1, 0, h - 1) * lineStride;
                    const float* qs = in + (size_t)psClampi(y - r, 0, h - 1) * lineStride;
                    for (int k = 0; k < m; ++k) sums[k] += (double)qa[k] - (double)qs[k];
                }
                std::swap(in, out);
            }
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < bw; ++x)
                    for (int c = 0; c < nch; ++c)
                        data[((size_t)y * w + x0 + x) * pstride + c] = in[(size_t)y * lineStride + (size_t)x * pstride + c];
        }
    });
}

// ---------------------------------------------------------------- erosion
// van Herk / Gil-Werman running minimum, window 2r+1, samples outside are ignored.
void minFilter1D(const float* in, float* out, int n, size_t stride, int r,
                 std::vector<float>& ext, std::vector<float>& g, std::vector<float>& hh)
{
    const int W = 2 * r + 1;
    const int L = n + 2 * r;
    const float INF = std::numeric_limits<float>::infinity();
    ext.resize(L); g.resize(L); hh.resize(L);
    for (int j = 0; j < L; ++j) ext[j] = (j >= r && j < n + r) ? in[(size_t)(j - r) * stride] : INF;
    for (int j = 0; j < L; ++j) g[j] = (j % W == 0) ? ext[j] : std::min(g[j - 1], ext[j]);
    for (int j = L - 1; j >= 0; --j) hh[j] = (j % W == W - 1 || j == L - 1) ? ext[j] : std::min(hh[j + 1], ext[j]);
    for (int i = 0; i < n; ++i) out[(size_t)i * stride] = std::min(hh[i], g[i + 2 * r]);
}

void erode(std::vector<float>& v, int w, int h, int r)
{
    if (r <= 0) return;
    std::vector<float> tmp(v.size());
    parallelFor(h, [&](int y0, int y1) {
        std::vector<float> e, g, hh;
        for (int y = y0; y < y1; ++y) minFilter1D(v.data() + (size_t)y * w, tmp.data() + (size_t)y * w, w, 1, r, e, g, hh);
    });
    // vertical: same van Herk / Gil-Werman scheme on blocks of 64 columns walked row by row
    const int block = 64;
    const int nBlocks = (w + block - 1) / block;
    parallelFor(nBlocks, [&](int b0, int b1) {
        const int W = 2 * r + 1, L = h + 2 * r;
        const float INF = std::numeric_limits<float>::infinity();
        std::vector<float> g((size_t)L * block), hh((size_t)L * block);
        for (int bi = b0; bi < b1; ++bi) {
            const int x0 = bi * block, bw = std::min(w, x0 + block) - x0;
            auto ext = [&](int j, int x) { return (j >= r && j < h + r) ? tmp[(size_t)(j - r) * w + x0 + x] : INF; };
            for (int j = 0; j < L; ++j)
                for (int x = 0; x < bw; ++x)
                    g[(size_t)j * block + x] = (j % W == 0) ? ext(j, x) : std::min(g[(size_t)(j - 1) * block + x], ext(j, x));
            for (int j = L - 1; j >= 0; --j)
                for (int x = 0; x < bw; ++x)
                    hh[(size_t)j * block + x] = (j % W == W - 1 || j == L - 1) ? ext(j, x)
                                                : std::min(hh[(size_t)(j + 1) * block + x], ext(j, x));
            for (int i = 0; i < h; ++i)
                for (int x = 0; x < bw; ++x)
                    v[(size_t)i * w + x0 + x] = std::min(hh[(size_t)i * block + x], g[(size_t)(i + 2 * r) * block + x]);
        }
    });
}

} // namespace

// ======================================================================= main
void processCPU(const Params& p, const float* src, const float* matte,
                const float* vec, float* dst)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0) return;
    const size_t N = (size_t)w * h;

    // 1. valid area
    std::vector<float> valid(N);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) valid[i] = computeValid(p, matte ? matte[i] : src[4 * i + 3]);
    });
    erode(valid, w, h, (int)std::lround(p.shrink));
    if (p.edgeSoftness > 0.f) gaussianish(valid, w, h, 1, 1, p.edgeSoftness);

    // 2. unpremultiplied source + weighted base
    std::vector<float> rgbU(4 * N), base(4 * N);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
            unpremultSource(p, src + 4 * i, &rgbU[4 * i]);
            buildBase(&rgbU[4 * i], valid[i], &base[4 * i]);
        }
    });

    // 3. nearest seed (exact distance transform) -> distance, Stretch colours
    std::vector<Seed> seedA;
    const bool needSeeds = p.spreadType != kTypeInterpolate || p.maxDistance > 0.f;
    const float seedThr = effectiveSeedThreshold(p);
    if (needSeeds) {
        seedA.resize(N);
        std::vector<int> nearestY(N);
        // pass 1: columns, scanned row by row on blocks of columns (cache friendly)
        const int block = 64;
        const int nBlocks = (w + block - 1) / block;
        parallelFor(nBlocks, [&](int b0, int b1) {
            std::vector<int> mark(block);
            for (int bi = b0; bi < b1; ++bi) {
                const int x0 = bi * block, x1 = std::min(w, x0 + block);
                std::fill(mark.begin(), mark.end(), -1);
                for (int y = 0; y < h; ++y)
                    for (int x = x0; x < x1; ++x) {
                        const size_t i = (size_t)y * w + x;
                        if (isSeed(valid[i], seedThr)) mark[x - x0] = y;
                        nearestY[i] = mark[x - x0];
                    }
                std::fill(mark.begin(), mark.end(), -1);
                for (int y = h - 1; y >= 0; --y)
                    for (int x = x0; x < x1; ++x) {
                        const size_t i = (size_t)y * w + x;
                        if (isSeed(valid[i], seedThr)) mark[x - x0] = y;
                        const int up = nearestY[i], next = mark[x - x0];
                        int best = up;
                        if (up < 0) best = next;
                        else if (next >= 0 && next - y < y - up) best = next;
                        nearestY[i] = best;
                    }
            }
        });
        // pass 2: rows, lower envelope of parabolas
        parallelFor(h, [&](int y0, int y1) {
            std::vector<int> v(w), zd(w);
            std::vector<long long> zn(w);
            for (int y = y0; y < y1; ++y)
                edtRow(nearestY.data() + (size_t)y * w, w, y, 1, v.data(), zn.data(), zd.data(), seedA.data() + (size_t)y * w);
        });
    }
    const Seed* seeds = seedA.data();
    std::vector<float> distField;
    if (needSeeds) {
        distField.resize(N);
        parallelFor(h, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) distField[(size_t)y * w + x] = seedDistance(seeds[(size_t)y * w + x], x, y);
        });
    }

    // 4. push-pull pyramid (Interpolate, and fallback of Vector Warp)
    std::vector<float> pushed;
    if (p.spreadType != kTypeStretch) {
        std::vector<std::vector<float>> levels;
        std::vector<int> lw, lh;
        // Vector Warp still needs `base` for its march; otherwise it can be moved, not copied
        if (p.spreadType == kTypeVectorWarp) levels.push_back(base);
        else levels.push_back(std::move(base));
        lw.push_back(w); lh.push_back(h);
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
            float* f = levels[l].data();
            const float* c = levels[l + 1].data();
            const int fw = lw[l], fh = lh[l], cw = lw[l + 1], ch = lh[l + 1];
            parallelFor(fh, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < fw; ++x) pushPixel(f + 4 * ((size_t)y * fw + x), c, cw, ch, x, y);
            });
        }
        pushed.swap(levels[0]);
    }

    // 5. fill colour + spread matte, 6. spread blur, 7. composite.
    //    Without spread blur, fill and composite are done in the same pass (no fill buffer).
    auto fillPixel = [&](int x, int y, float* o) {
        const size_t i = (size_t)y * w + x;
        Seed s; s.x = -1; s.y = -1;
        if (needSeeds) s = seeds[i];
        const bool hasSeed = needSeeds && s.x >= 0;
        const float dist = hasSeed ? distField[i] : 0.f;
        bool hasFill = false;
        if (p.spreadType == kTypeStretch) {
            hasFill = hasSeed;
            if (hasSeed) {
                const float* c = &rgbU[4 * ((size_t)s.y * w + s.x)];
                o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
            } else { o[0] = o[1] = o[2] = 0.f; }
        } else {
            hasFill = resolveWeighted(&pushed[4 * i], o);
            if (p.spreadType == kTypeVectorWarp && valid[i] < 1.f) {
                float dx, dy, C[3];
                spreadVector(p, vec ? vec + 4 * i : nullptr, &dx, &dy);
                const float A = marchPixel(p, base.data(), distField.data(), w, h, x, y, dx, dy, C);
                if (A > 0.f) {
                    const float k = 1.f - A;
                    o[0] = C[0] + k * o[0]; o[1] = C[1] + k * o[1]; o[2] = C[2] + k * o[2];
                    hasFill = true;
                }
            }
        }
        o[3] = spreadMatte(p, hasFill, hasSeed, dist);
    };

    if (p.blur <= 0.f) {
        parallelFor(h, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t)y * w + x;
                    float f4[4];
                    fillPixel(x, y, f4);
                    compositePixel(p, &rgbU[4 * i], valid[i], f4, f4[3], dst + 4 * i);
                }
        });
        return;
    }

    std::vector<float> fill(4 * N);
    parallelFor(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) fillPixel(x, y, &fill[4 * ((size_t)y * w + x)]);
    });
    gaussianish(fill, w, h, 4, 3, p.blur);
    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i)
            compositePixel(p, &rgbU[4 * i], valid[i], &fill[4 * i], fill[4 * i + 3], dst + 4 * i);
    });
}

} // namespace pushit
