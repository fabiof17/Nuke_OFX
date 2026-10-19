// GlowCPU.cpp
// Multithreaded CPU back-end of expo_GLOW_OFX. Same pipeline as GlowCUDA.cu.
#include "GlowCore.h"

#include <cmath>
#include <memory>
#include <thread>

namespace expoglow {
namespace {

// Runs f(begin, end) over [0, n) split across hardware threads. `work` is the total amount
// of work (pixels): small jobs run on the calling thread, threads would cost more than they save.
template <class F>
void parallelFor(int n, F&& f, size_t work = (size_t)-1)
{
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 16 || nt == 1 || work < 65536) { f(0, n); return; }
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

// out[i] = sum_t k[t] * srcs[t][i] for i in [0, len). The sum of a block of 16 values is
// kept in registers across all the taps (one store per output value instead of one per tap).
inline void macRows(float* __restrict out, size_t len, const float* const* srcs, const float* k, int taps)
{
    constexpr size_t B = 16;
    size_t b = 0;
    for (; b + B <= len; b += B) {
        float acc[B] = {};
        for (int t = 0; t < taps; ++t) {
            const float kt = k[t];
            const float* q = srcs[t] + b;
            for (size_t i = 0; i < B; ++i) acc[i] += kt * q[i];
        }
        for (size_t i = 0; i < B; ++i) out[b + i] = acc[i];
    }
    for (; b < len; ++b) {
        float a = 0.f;
        for (int t = 0; t < taps; ++t) a += k[t] * srcs[t][b];
        out[b] = a;
    }
}

// Horizontal merged-kernel convolution of one RGBA row (edge pixels repeated).
inline void convRow(const float* row, float* out, int w, const float* kk, int taps)
{
    const int c = taps / 2;
    const int xa = std::min(c, w), xb = std::max(xa, w - c);
    for (int x = 0; x < xa; ++x) convPixel(row, w, 4, x, kk, taps, out + 4 * x);
    for (int x = xb; x < w; ++x) convPixel(row, w, 4, x, kk, taps, out + 4 * x);
    if (xb > xa) {
        const float* srcs[kMaxTaps];
        for (int t = 0; t < taps; ++t) srcs[t] = row + 4 * (xa + t - c);
        macRows(out + 4 * xa, 4 * (size_t)(xb - xa), srcs, kk, taps);
    }
}

// Vertical merged-kernel convolution producing row y of an image whose rows are given by
// rowPtr(yy) for yy in [0, h) (edge rows repeated).
template <class RowPtr>
inline void convCol(RowPtr rowPtr, int y, int h, float* out, int w, const float* kk, int taps)
{
    const int c = taps / 2;
    const float* srcs[kMaxTaps];
    for (int t = 0; t < taps; ++t) srcs[t] = rowPtr(egClampi(y + t - c, h));
    macRows(out, 4 * (size_t)w, srcs, kk, taps);
}

// Vertical fractional box on columns [x0, x1), walking rows so memory stays sequential.
// Same arithmetic as boxLine() (double sums, edge pixels repeated outside).
void boxColumns(const float* in, float* out, int w, int h, int x0, int x1, float r, std::vector<double>& sums)
{
    const int   m = (int)std::floor(r);
    const float a = r - (float)m;
    const double norm = 1.0 / (2.0 * (double)r + 1.0);
    const int bw = x1 - x0;
    sums.assign((size_t)3 * bw, 0.0);
    auto px = [&](int y, int x) { return in + 4 * ((size_t)egClampi(y, h) * w + x); };
    for (int j = -m; j <= m; ++j)
        for (int x = 0; x < bw; ++x) {
            const float* q = px(j, x0 + x);
            sums[3 * x] += q[0]; sums[3 * x + 1] += q[1]; sums[3 * x + 2] += q[2];
        }
    for (int i = 0; i < h; ++i) {
        for (int x = 0; x < bw; ++x) {
            double* s = &sums[3 * x];
            const float* ql = px(i - m - 1, x0 + x);
            const float* qr = px(i + m + 1, x0 + x);
            float* o = out + 4 * ((size_t)i * w + x0 + x);
            o[0] = (float)((s[0] + a * ((double)ql[0] + qr[0])) * norm);
            o[1] = (float)((s[1] + a * ((double)ql[1] + qr[1])) * norm);
            o[2] = (float)((s[2] + a * ((double)ql[2] + qr[2])) * norm);
            o[3] = 0.f;
            const float* qo = px(i - m, x0 + x);
            s[0] += (double)qr[0] - qo[0]; s[1] += (double)qr[1] - qo[1]; s[2] += (double)qr[2] - qo[2];
        }
    }
}

} // namespace

// Area average of a w x h image reduced by f, computed from an image already reduced by g
// (g divides f). Partial blocks at the edges are weighted by their real pixel counts, so the
// result is exactly the area average of the full-resolution image.
static void downsampleFrom(const float* src, int sw, int sh, int g, int w, int h, int f, float* out, int ws, int hs)
{
    const int m = f / g;
    parallelFor(hs, [&](int y0, int y1) {
        for (int Y = y0; Y < y1; ++Y) {
            const int ly0 = Y * m, ly1 = std::min(sh, ly0 + m);
            for (int X = 0; X < ws; ++X) {
                const int kx0 = X * m, kx1 = std::min(sw, kx0 + m);
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, cnt = 0.0;
                for (int l = ly0; l < ly1; ++l) {
                    const double cy = (double)(std::min(h, (l + 1) * g) - l * g);
                    const float* q = src + 4 * ((size_t)l * sw + kx0);
                    for (int k = kx0; k < kx1; ++k, q += 4) {
                        const double c = cy * (double)(std::min(w, (k + 1) * g) - k * g);
                        s0 += c * q[0]; s1 += c * q[1]; s2 += c * q[2]; cnt += c;
                    }
                }
                float* o = out + 4 * ((size_t)Y * ws + X);
                const double inv = cnt > 0.0 ? 1.0 / cnt : 0.0;
                o[0] = (float)(s0 * inv); o[1] = (float)(s1 * inv); o[2] = (float)(s2 * inv); o[3] = 0.f;
            }
        }
    }, (size_t)ws * hs * m * m);
}

void computeGlowCPU(const PrepParams& p, const std::vector<StepPlan>& plan,
                    const float* src, const float* area, float* glow)
{
    const int w = p.width, h = p.height;
    if (w <= 0 || h <= 0) return;
    const size_t N = (size_t)w * h;

    // uninitialised: every value is written by prepPixel() just below
    std::unique_ptr<float[]> prepMem(new float[4 * N]);
    float* const prep = prepMem.get();
    std::vector<float> bufA, bufB, padded;

    parallelFor(h, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i)
            prepPixel(p, src + 4 * i, area ? area[i] : 1.f, prep + 4 * i);
    }, N);

    // ---- generic blur (used for reduced images and for long cascades at full resolution)
    auto blur = [&](const float* in, int bw, int bh, const StepPlan& step, std::vector<float>& out) {
        out.resize(4 * (size_t)bw * bh);
        const size_t work = (size_t)bw * bh;
        if (step.taps > 0) {
            // Merged kernel, one pass per direction: repeating the edge pixels (clamped indices)
            // is already the exact extension, no padding needed.
            const int taps = step.taps;
            const float* kk = step.kernel;
            if (bufA.size() < out.size()) bufA.resize(out.size());
            float* tmp = bufA.data();
            parallelFor(bh, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) convRow(in + 4 * (size_t)y * bw, tmp + 4 * (size_t)y * bw, bw, kk, taps);
            }, work * taps);
            float* dst = out.data();
            parallelFor(bh, [&](int y0, int y1) {
                auto rowPtr = [&](int yy) { return (const float*)(tmp + 4 * (size_t)yy * bw); };
                for (int y = y0; y < y1; ++y) convCol(rowPtr, y, bh, dst + 4 * (size_t)y * bw, bw, kk, taps);
            }, work * taps);
            return;
        }
        // Long cascade: pad with the repeated edges by the cascade reach, filter, crop.
        const int pad = planPadding(step);
        const int pw = bw + 2 * pad, ph = bh + 2 * pad;
        const size_t n = 4 * (size_t)pw * ph, pwork = (size_t)pw * ph;
        padded.resize(n);
        if (bufA.size() < n) bufA.resize(n);
        if (bufB.size() < n) bufB.resize(n);
        parallelFor(ph, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < pw; ++x) padPixel(in, bw, bh, pad, x, y, &padded[4 * ((size_t)y * pw + x)]);
        }, pwork);
        const float* cur = padded.data();
        float* outs[2] = { bufA.data(), bufB.data() };
        int k = 0;
        for (int pass = 0; pass < step.passes; ++pass) {
            float* o = outs[k++ & 1];
            const float r = step.radius[pass];
            parallelFor(ph, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    boxLine(cur + 4 * (size_t)y * pw, o + 4 * (size_t)y * pw, pw, 4, r);
            }, pwork * 4);
            cur = o;
        }
        const int block = 32;
        const int nBlocks = (pw + block - 1) / block;
        for (int pass = 0; pass < step.passes; ++pass) {
            float* o = outs[k++ & 1];
            const float r = step.radius[pass];
            parallelFor(nBlocks, [&](int b0, int b1) {
                std::vector<double> sums;
                for (int b = b0; b < b1; ++b)
                    boxColumns(cur, o, pw, ph, b * block, std::min(pw, (b + 1) * block), r, sums);
            }, pwork * 4);
            cur = o;
        }
        parallelFor(bh, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                std::copy(cur + 4 * ((size_t)(y + pad) * pw + pad), cur + 4 * ((size_t)(y + pad) * pw + pad + bw),
                          out.begin() + 4 * (size_t)y * bw);
        }, work);
    };

    // ---- 1. reduced steps (Nuke "quality"): hierarchical area averages, then blur
    struct Reduced { int f, ws, hs; const StepPlan* step; std::vector<float> img, blurred;
                     std::vector<int> u0, u1; std::vector<float> fu; };
    std::vector<Reduced> reduced;
    for (const StepPlan& step : plan) {
        if (step.factor <= 1) continue;
        Reduced r;
        r.f = step.factor; r.ws = (w + r.f - 1) / r.f; r.hs = (h + r.f - 1) / r.f; r.step = &step;
        // source: the most reduced level already built whose factor divides f, else full res
        const Reduced* from = nullptr;
        for (const Reduced& q : reduced)
            if (r.f % q.f == 0 && (!from || q.f > from->f)) from = &q;
        r.img.resize(4 * (size_t)r.ws * r.hs);
        if (from && from->f == r.f) r.img = from->img;
        else if (from) downsampleFrom(from->img.data(), from->ws, from->hs, from->f, w, h, r.f, r.img.data(), r.ws, r.hs);
        else           downsampleFrom(prep, w, h, 1, w, h, r.f, r.img.data(), r.ws, r.hs);
        blur(r.img.data(), r.ws, r.hs, step, r.blurred);
        // per-column bilinear tables for the final upscale
        r.u0.resize(w); r.u1.resize(w); r.fu.resize(w);
        for (int x = 0; x < w; ++x) {
            float u = ((float)x + 0.5f) / (float)r.f - 0.5f;
            u = u < 0.f ? 0.f : (u > (float)(r.ws - 1) ? (float)(r.ws - 1) : u);
            r.u0[x] = (int)u; r.u1[x] = r.u0[x] + 1 < r.ws ? r.u0[x] + 1 : r.u0[x]; r.fu[x] = u - (float)r.u0[x];
        }
        reduced.push_back(std::move(r));
    }

    // ---- 2. long cascades at full resolution (quality 0 only): separate path
    std::vector<const StepPlan*> fullres;
    std::vector<float> tmpBlur;
    bool anyCascade = false;
    for (const StepPlan& step : plan) {
        if (step.factor > 1) continue;
        if (step.taps > 0) { fullres.push_back(&step); continue; }
        if (!anyCascade) { std::fill(glow, glow + 4 * N, 0.f); anyCascade = true; }
        blur(prep, w, h, step, tmpBlur);
        parallelFor(h, [&](int y0, int y1) {
            for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i)
                accumulatePixel(glow + 4 * i, &tmpBlur[4 * i], step.weight);
        }, N);
    }

    // ---- 3. single final pass, by strips of rows so that intermediate rows stay in cache:
    //         full-res steps (horizontal then vertical merged kernels) + bilinear upscale of
    //         the reduced steps, each weighted and summed into the glow.
    const int strip = 32;
    const int nStrips = (h + strip - 1) / strip;
    int maxC = 0;
    for (const StepPlan* st : fullres) maxC = std::max(maxC, st->taps / 2);
    parallelFor(nStrips, [&](int s0, int s1) {
        std::vector<float> hrows(4 * (size_t)w * (strip + 2 * maxC)), vrow(4 * (size_t)w), rowV;
        for (int sIdx = s0; sIdx < s1; ++sIdx) {
            const int ya = sIdx * strip, yb = std::min(h, ya + strip);
            if (!anyCascade)
                std::fill(glow + 4 * (size_t)ya * w, glow + 4 * (size_t)yb * w, 0.f);
            for (const StepPlan* st : fullres) {
                const int taps = st->taps, c = taps / 2;
                const float* kk = st->kernel;
                // horizontal pass on the rows this strip needs (clamped to the image)
                const int ha = std::max(0, ya - c), hb = std::min(h, yb + c);
                for (int y = ha; y < hb; ++y)
                    convRow(prep + 4 * (size_t)y * w, hrows.data() + 4 * (size_t)(y - ha) * w, w, kk, taps);
                // vertical pass + weighted accumulation
                auto rowPtr = [&](int yy) { return (const float*)(hrows.data() + 4 * (size_t)(yy - ha) * w); };
                for (int y = ya; y < yb; ++y) {
                    convCol(rowPtr, y, h, vrow.data(), w, kk, taps);
                    // the source was sanitised in prepPixel(): no NaN/inf can appear here
                    const float wt4[4] = { st->weight[0], st->weight[1], st->weight[2], 0.f };
                    float* __restrict g = glow + 4 * (size_t)y * w;
                    const float* __restrict vr = vrow.data();
                    for (int x = 0; x < w; ++x)
                        for (int k = 0; k < 4; ++k) g[4 * x + k] += vr[4 * x + k] * wt4[k];
                }
            }
            for (int y = ya; y < yb; ++y) {
                float* g = glow + 4 * (size_t)y * w;
                for (const Reduced& r : reduced) {
                    float v = ((float)y + 0.5f) / (float)r.f - 0.5f;
                    v = v < 0.f ? 0.f : (v > (float)(r.hs - 1) ? (float)(r.hs - 1) : v);
                    const int v0 = (int)v, v1 = v0 + 1 < r.hs ? v0 + 1 : v0;
                    const float fv = v - (float)v0;
                    const float* ra = r.blurred.data() + 4 * (size_t)v0 * r.ws;
                    const float* rc = r.blurred.data() + 4 * (size_t)v1 * r.ws;
                    rowV.resize(4 * (size_t)r.ws);
                    for (int i = 0; i < 4 * r.ws; ++i) rowV[i] = ra[i] + (rc[i] - ra[i]) * fv;
                    const float wt4[4] = { r.step->weight[0], r.step->weight[1], r.step->weight[2], 0.f };
                    float* __restrict a = g;
                    const float* rv = rowV.data();
                    for (int x = 0; x < w; ++x) {
                        const float* l = rv + 4 * r.u0[x];
                        const float* rr = rv + 4 * r.u1[x];
                        const float fu = r.fu[x];
                        for (int k = 0; k < 4; ++k) a[4 * x + k] += (l[k] + (rr[k] - l[k]) * fu) * wt4[k];
                    }
                }
            }
        }
    }, N * 16);
}

} // namespace expoglow
