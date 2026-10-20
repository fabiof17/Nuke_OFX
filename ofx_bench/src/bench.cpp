// ofx_bench: standalone CPU / GPU benchmark of the push_IT_OFX, apGlow_OFX, L_Despill_OFX, despill_IT_OFX and
// MasterKeyer_OFX cores.
// Runs each test on the CPU and on the CUDA GPU, prints the timings and checks that both
// give the same image. No Nuke needed.
#include "push_it/push_IT_OFXCore.h"
#include "expo_glow/GlowCore.h"
#include "l_despill/DespillCore.h"
#include "despill_it/DespillITCore.h"
#include "master_keyer/MasterKeyerCore.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

std::string transferTest(size_t bytes, int runs);

namespace {

using Clock = std::chrono::steady_clock;

double bestOf(int runs, const std::function<bool()>& f)
{
    double best = 1e30;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = Clock::now();
        if (!f()) return -1.0;
        best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return best;
}

double maxDiff(const std::vector<float>& a, const std::vector<float>& b, double& peak)
{
    double d = 0.0;
    peak = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, (double)std::fabs(a[i] - b[i]));
        peak = std::max(peak, (double)std::fabs(a[i]));
    }
    return d;
}

// Test image: discs, a bar, colour gradients, sparse hot pixels.
void makeImage(int w, int h, std::vector<float>& rgba, std::vector<float>& plate, std::vector<float>& matte)
{
    rgba.assign((size_t)4 * w * h, 0.f);
    plate.assign((size_t)4 * w * h, 0.f);
    matte.assign((size_t)w * h, 0.f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            const float dx = x - w * 0.5f, dy = y - h * 0.5f, r = std::sqrt(dx * dx + dy * dy);
            float a = std::fmin(1.f, std::fmax(0.f, (h * 0.3f - r) / 2.f));
            if (std::fabs(dy) < h * 0.04f && std::fabs(dx) < w * 0.4f) a = 1.f;
            const float cr = 0.5f + 0.5f * std::sin(x * 0.02f), cg = 0.5f + 0.5f * std::cos(y * 0.03f),
                        cb = 0.5f + 0.5f * std::sin((x + y) * 0.01f);
            const float hot = ((x * 7919 + y * 104729) % 997 == 0) ? 20.f : 1.f;
            rgba[4 * i] = cr * a * hot; rgba[4 * i + 1] = cg * a * hot; rgba[4 * i + 2] = cb * a * hot; rgba[4 * i + 3] = a;
            plate[4 * i] = cr; plate[4 * i + 1] = cg; plate[4 * i + 2] = cb; plate[4 * i + 3] = 1.f;
            matte[i] = std::fmin(1.f, std::fmax(0.f, (h * 0.15f - r) / 2.f));
        }
}

void printRow(const char* name, double cpu, double gpu, double diff, double peak)
{
    char g[32], s[32], d[48];
    if (gpu < 0) { std::snprintf(g, sizeof g, "   n/a"); std::snprintf(s, sizeof s, "   -"); std::snprintf(d, sizeof d, "GPU unavailable"); }
    else {
        std::snprintf(g, sizeof g, "%6.1f", gpu);
        std::snprintf(s, sizeof s, "x%5.1f", cpu / std::max(gpu, 1e-3));
        const double rel = peak > 0 ? diff / peak : 0.0;
        std::snprintf(d, sizeof d, "%s (%.1e)", rel < 1e-4 ? "identical" : "DIFFERENT", rel);
    }
    std::printf("  %-34s %8.1f ms %8s ms %8s   %s\n", name, cpu, g, s, d);
}

} // namespace

int main(int argc, char** argv)
{
    const int runs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    std::string info;
    const bool gpuOk = pushit::cudaDeviceInfo(info);
    std::printf("ofx_bench - CPU threads: %u - %s\n", std::thread::hardware_concurrency(), info.c_str());
    std::printf("Best of %d runs. GPU times include the upload/download, as in Nuke.\n", runs);

    const int sizes[2][2] = { { 1920, 1080 }, { 3840, 2160 } };
    for (const auto& sz : sizes) {
        const int w = sz[0], h = sz[1];
        std::vector<float> src, plate, matte, outC((size_t)4 * w * h), outG((size_t)4 * w * h);
        makeImage(w, h, src, plate, matte);
        std::printf("\n=== %dx%d ===\n  %-34s %11s %11s %8s   %s\n", w, h, "test", "CPU", "GPU", "speed-up", "CPU vs GPU");

        // ---------------- push_IT_OFX
        struct PCase { const char* name; pushit::Params p; bool plateInput; };
        std::string profiles;
        std::vector<PCase> pcs;
        pushit::Params b; b.width = w; b.height = h;
        { pushit::Params p = b; pcs.push_back({ "push_IT  Interpolate", p, false }); }
        { pushit::Params p = b; p.spreadType = pushit::kTypeStretch; pcs.push_back({ "push_IT  Stretch", p, false }); }
        { pushit::Params p = b; p.spreadType = pushit::kTypeStretch; p.maxDistance = 100; p.falloff = 30; p.blur = 50;
          pcs.push_back({ "push_IT  Stretch + MaxDist + Blur50", p, false }); }
        { pushit::Params p = b; p.spreadType = pushit::kTypeVectorWarp; p.vectorSource = pushit::kVectorConstant;
          p.constX = 0; p.constY = -300; pcs.push_back({ "push_IT  Vector Warp 300px", p, false }); }
        { pushit::Params p = b; p.spreadMode = pushit::kModeContract; p.shrink = 5; p.edgeSoftness = 5;
          pcs.push_back({ "push_IT  Contract shrink5 soft5", p, true }); }
        { pushit::Params p = b; p.blur = 500; pcs.push_back({ "push_IT  Interpolate + Blur500", p, false }); }

        for (const PCase& c : pcs) {
            const float* s = c.plateInput ? plate.data() : src.data();
            const float* m = c.plateInput ? matte.data() : nullptr;
            const double cpu = bestOf(runs, [&] { pushit::processCPU(c.p, s, m, nullptr, outC.data()); return true; });
            double gpu = -1, diff = 0, peak = 0;
            if (gpuOk) {
                std::string err;
                gpu = bestOf(runs, [&] { return pushit::processCUDA(c.p, s, m, nullptr, outG.data(), err); });
                if (gpu >= 0) diff = maxDiff(outC, outG, peak);
                else std::printf("  GPU error: %s\n", err.c_str());
            }
            printRow(c.name, cpu, gpu, diff, peak);
            if (gpuOk && gpu >= 0) {
                std::string err;
                pushit::cudaSetProfiling(true);
                const auto t0 = Clock::now();
                pushit::processCUDA(c.p, s, m, nullptr, outG.data(), err);
                const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                pushit::cudaSetProfiling(false);
                char head[160];
                std::snprintf(head, sizeof head, "    %s  (wall clock %.2f ms)\n", c.name, wall);
                profiles += head + pushit::cudaLastProfile();
            }
        }

        // ---------------- apGlow_OFX
        struct GCase { const char* name; float maxSize; int filter; int steps; };
        const GCase gcs[] = {
            { "apGlow 2000px gaussian 10 steps", 2000, expoglow::kFilterGaussian, 10 },
            { "apGlow 2000px box 10 steps",      2000, expoglow::kFilterBox,      10 },
            { "apGlow 300px gaussian 20 steps",   300, expoglow::kFilterGaussian, 20 },
        };
        expoglow::PrepParams pp; pp.width = w; pp.height = h;
        for (const GCase& c : gcs) {
            expoglow::GlowSettings gs;
            gs.maxSize = c.maxSize; gs.filter = c.filter; gs.steps = c.steps;
            const std::vector<expoglow::StepPlan> plan = expoglow::planGlow(gs);
            const double cpu = bestOf(runs, [&] { expoglow::computeGlowCPU(pp, plan, src.data(), nullptr, outC.data()); return true; });
            double gpu = -1, diff = 0, peak = 0;
            if (gpuOk) {
                std::string err;
                gpu = bestOf(runs, [&] { return expoglow::computeGlowCUDA(pp, plan, src.data(), nullptr, outG.data(), err); });
                if (gpu >= 0) diff = maxDiff(outC, outG, peak);
                else std::printf("  GPU error: %s\n", err.c_str());
            }
            printRow(c.name, cpu, gpu, diff, peak);
            if (gpuOk && gpu >= 0) {
                std::string err;
                expoglow::cudaSetProfiling(true);
                const auto t0 = Clock::now();
                expoglow::computeGlowCUDA(pp, plan, src.data(), nullptr, outG.data(), err);
                const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                expoglow::cudaSetProfiling(false);
                char head[160];
                std::snprintf(head, sizeof head, "    %s  (wall clock %.2f ms)\n", c.name, wall);
                profiles += head + expoglow::cudaLastProfile();
            }
        }
        // ---------------- L_Despill_OFX (source buffer = image + margin, edges repeated)
        struct DCase { const char* name; int median, clean; float blur; bool useBG; };
        const DCase dcs[] = {
            { "L_Despill default (log average)", 0, 0, 0.f, false },
            { "L_Despill median5 clean4 blur20 +BG", 5, 4, 20.f, true },
        };
        std::vector<float> bgRGB((size_t)3 * w * h);
        for (size_t i = 0; i < (size_t)w * h; ++i) for (int k = 0; k < 3; ++k) bgRGB[3 * i + k] = plate[4 * i + k];
        for (const DCase& c : dcs) {
            ldespill::Params dp;
            dp.medianRadius = c.median; dp.cleanRadius = c.clean; dp.matteBlur = c.blur; dp.hasBG = c.useBG;
            const int m = ldespill::spatialMargin(dp);
            dp.bufW = w + 2 * m; dp.bufH = h + 2 * m; dp.winX = m; dp.winY = m; dp.winW = w; dp.winH = h;
            std::vector<float> dsrc((size_t)4 * dp.bufW * dp.bufH);
            for (int y = 0; y < dp.bufH; ++y)
                for (int x = 0; x < dp.bufW; ++x) {
                    const int sx = std::min(std::max(x - m, 0), w - 1), sy = std::min(std::max(y - m, 0), h - 1);
                    for (int k = 0; k < 4; ++k) dsrc[4 * ((size_t)y * dp.bufW + x) + k] = src[4 * ((size_t)sy * w + sx) + k];
                }
            const float* bgp = c.useBG ? bgRGB.data() : nullptr;
            const double cpu = bestOf(runs, [&] { ldespill::processCPU(dp, dsrc.data(), bgp, nullptr, outC.data()); return true; });
            double gpu = -1, diff = 0, peak = 0;
            if (gpuOk) {
                std::string err;
                gpu = bestOf(runs, [&] { return ldespill::processCUDA(dp, dsrc.data(), bgp, nullptr, outG.data(), err); });
                if (gpu >= 0) diff = maxDiff(outC, outG, peak);
                else std::printf("  GPU error: %s\n", err.c_str());
            }
            printRow(c.name, cpu, gpu, diff, peak);
            if (gpuOk && gpu >= 0) {
                std::string err;
                ldespill::cudaSetProfiling(true);
                const auto t0 = Clock::now();
                ldespill::processCUDA(dp, dsrc.data(), bgp, nullptr, outG.data(), err);
                const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                ldespill::cudaSetProfiling(false);
                char head[160];
                std::snprintf(head, sizeof head, "    %s  (wall clock %.2f ms)\n", c.name, wall);
                profiles += head + ldespill::cudaLastProfile();
            }
        }

        // ---------------- despill_IT_OFX
        {
            std::vector<float> bg3((size_t)3 * w * h);
            for (size_t i = 0; i < (size_t)w * h; ++i) for (int k = 0; k < 3; ++k) bg3[3 * i + k] = plate[4 * i + k];
            struct ICase { const char* name; bool bg; };
            const ICase ics[] = { { "despill_IT default", false }, { "despill_IT + BG light + wrap", true } };
            for (const ICase& c : ics) {
                dspit::Params ip; ip.width = w; ip.height = h;
                dspit::Inputs in; in.src = src.data();
                if (c.bg) { ip.hasBG = true; ip.spillLight = dspit::kLightBackground; ip.wrapAmount = 0.5f; ip.bgBlur = 30.f; in.bg = bg3.data(); }
                const double cpu = bestOf(runs, [&] { dspit::processCPU(ip, in, outC.data()); return true; });
                double gpu = -1, diff = 0, peak = 0;
                if (gpuOk) {
                    std::string err;
                    gpu = bestOf(runs, [&] { return dspit::processCUDA(ip, in, outG.data(), err); });
                    if (gpu >= 0) diff = maxDiff(outC, outG, peak);
                    else std::printf("  GPU error: %s\n", err.c_str());
                }
                printRow(c.name, cpu, gpu, diff, peak);
                if (gpuOk && gpu >= 0) {
                    std::string err;
                    dspit::cudaSetProfiling(true);
                    const auto t0 = Clock::now();
                    dspit::processCUDA(ip, in, outG.data(), err);
                    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    dspit::cudaSetProfiling(false);
                    char head[160];
                    std::snprintf(head, sizeof head, "    %s  (wall clock %.2f ms)\n", c.name, wall);
                    profiles += head + dspit::cudaLastProfile();
                }
            }
        }

        // ---------------- MasterKeyer_OFX (green screen plate made from the test image)
        {
            std::vector<float> gs((size_t)4 * w * h), bg3((size_t)3 * w * h);
            const float S[3] = { 0.08f, 0.45f, 0.12f };
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t)y * w + x;
                    const float a = src[4 * i + 3], g = 0.8f + 0.4f * x / w;
                    for (int k = 0; k < 3; ++k) {
                        gs[4 * i + k] = a * plate[4 * i + k] + (1.f - a) * S[k] * g;
                        bg3[3 * i + k] = plate[4 * ((size_t)(h - 1 - y) * w + x) + k];
                    }
                    gs[4 * i + 3] = 1.f;
                }
            struct KCase { const char* name; int mode; };
            const KCase kcs[] = { { "MasterKeyer default", 0 }, { "MasterKeyer local+degrain+blend+BG", 1 },
                                  { "MasterKeyer Combined KeyIn patches", 2 } };
            std::vector<float> key3((size_t)3 * w * h);
            for (size_t i = 0; i < (size_t)w * h; ++i) for (int k = 0; k < 3; ++k) key3[3 * i + k] = gs[4 * i + k];
            for (const KCase& c : kcs) {
                mk::Params kp; kp.width = w; kp.height = h;
                mk::Inputs in; in.src = gs.data();
                if (c.mode == 2) {
                    in.keyIn = key3.data();
                    kp.shadows = 0.5f; kp.midtones = -0.3f; kp.highlights = -1.f; kp.mix = 0.5f; kp.gain = 1.2f;
                    kp.patch[0].type = mk::kPatchWhite;
                    kp.patch[1].type = mk::kPatchBlack;
                    kp.patch[2].type = mk::kPatchEdge;
                    for (int q = 0; q < 3; ++q) {
                        kp.patch[0].lo[q] = 0.6f; kp.patch[0].hi[q] = 0.9f;
                        kp.patch[1].lo[q] = 0.0f; kp.patch[1].hi[q] = 0.1f;
                        kp.patch[2].lo[q] = 0.3f; kp.patch[2].hi[q] = 0.5f;
                    }
                    kp.spillColour[2] = 0.25f;
                    kp.matteMode = mk::kMatteCombined;
                    kp.output = mk::kOutFrontPremult;
                }
                if (c.mode == 1) {
                    kp.screenSource = mk::kScreenLocal; kp.degrain = true; kp.shrink = -2; kp.matteBlur = 2.f;
                    kp.edgeSize = 10.f; kp.edgeFromBG = 0.5f; kp.bgBlur = 30.f; kp.hasBG = true; in.bg = bg3.data();
                }
                const double cpu = bestOf(runs, [&] { mk::processCPU(kp, in, outC.data()); return true; });
                double gpu = -1, diff = 0, peak = 0;
                if (gpuOk) {
                    std::string err;
                    gpu = bestOf(runs, [&] { return mk::processCUDA(kp, in, outG.data(), err); });
                    if (gpu >= 0) diff = maxDiff(outC, outG, peak);
                    else std::printf("  GPU error: %s\n", err.c_str());
                }
                printRow(c.name, cpu, gpu, diff, peak);
                if (gpuOk && gpu >= 0) {
                    std::string err;
                    mk::cudaSetProfiling(true);
                    const auto t0 = Clock::now();
                    mk::processCUDA(kp, in, outG.data(), err);
                    const double wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    mk::cudaSetProfiling(false);
                    char head[160];
                    std::snprintf(head, sizeof head, "    %s  (wall clock %.2f ms)\n", c.name, wall);
                    profiles += head + mk::cudaLastProfile();
                }
            }
        }

        if (gpuOk) {
            std::printf("\n  GPU detail per stage (%dx%d):\n%s", w, h, profiles.c_str());
            std::printf("\n  Transfer speed for one RGBA float image (%.0f MB):\n%s", 16.0 * w * h / 1e6,
                        transferTest((size_t)16 * w * h, runs).c_str());
        }
    }
    pushit::cudaReleaseResources();
    expoglow::cudaReleaseResources();
    ldespill::cudaReleaseResources();
    dspit::cudaReleaseResources();
    mk::cudaReleaseResources();
    return 0;
}
