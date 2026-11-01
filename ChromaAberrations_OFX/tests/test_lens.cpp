// ChromaAberrations_OFX - validation tests of the real lenses (step 4)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Reference: Optiland (independent ray tracer, MIT licence), see tests/reference.
//   test_lens <folder of tests/reference>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "psfbank.h"
#include "reallens.h"
#include "render.h"

using namespace ca;

static int gFail = 0, gChecks = 0;
#define CHECK(cond, ...) do { ++gChecks; bool ok_ = (cond); std::printf("  [%s] ", ok_ ? " ok " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); if (!ok_) ++gFail; } while (0)

static std::string gRef = "tests/reference";
static const double kPi = 3.14159265358979323846;

static void testGlasses()
{
    std::printf("Glass catalogue (%d glasses) against Optiland (refractiveindex.info data)\n", glassCatalogSize());
    std::ifstream f(gRef + "/optiland_glass.txt");
    std::string line;
    double worst = 0.;
    int n = 0, missing = 0;
    while ( std::getline(f, line) ) {
        std::istringstream s(line);
        std::string name;
        s >> name;
        Glass g;
        if ( !findGlass(name, g) ) { ++missing; continue; }
        const double w[5] = { 400., 486.1327, 587.6, 656.2725, 700. };
        for (int i = 0; i < 5; ++i) {
            double r;
            s >> r;
            worst = std::max(worst, std::fabs(g.index(w[i]) - r));
        }
        ++n;
    }
    CHECK(n >= 19 && missing == 0, "%d glasses of the built-in lenses found (%d missing)", n, missing);
    CHECK(worst < 1e-6, "refractive index within %.1e of Optiland, 400 to 700 nm", worst);
    Glass m = modelGlass(1.5168, 64.17), bk7;
    findGlass("N-BK7", bk7);
    CHECK(std::fabs(m.nd() - 1.5168) < 1e-6 && std::fabs(m.vd() - 64.17) < 1e-3 && std::fabs(m.index(450.) - bk7.index(450.)) < 5e-4,
          "model glass 1.5168/64.17: nd %.5f, vd %.2f, n(450) %.5f (N-BK7 %.5f)", m.nd(), m.vd(), m.index(450.), bk7.index(450.));
}

struct RefRow { std::string lens, kind; double field, pupil, wl, value; };

static void testRayTracing()
{
    std::printf("Ray tracing against Optiland (built-in lenses at the prescription focus, 3 wavelengths, 4 fields)\n");
    std::ifstream f(gRef + "/optiland_reference.csv");
    std::string line;
    std::getline(f, line);
    std::vector<RefRow> rows;
    std::map<std::string, double> unit;
    while ( std::getline(f, line) ) {
        std::stringstream s(line);
        std::string a[6];
        for (int i = 0; i < 6; ++i) std::getline(s, a[i], ',');
        RefRow r = { a[0], a[1], std::atof(a[2].c_str()), std::atof(a[3].c_str()), std::atof(a[4].c_str()),
                     a[5] == "nan" ? std::nan("") : std::atof(a[5].c_str()) };
        if (r.kind == "unit") unit[r.lens] = r.value;
        else rows.push_back(r);
    }
    CHECK(unit.size() == 5, "reference data for %d lenses", (int)unit.size());
    std::map<std::string, RealLens> lenses;
    std::map<std::string, double> pixel;
    for (auto& kv : unit) {
        LensPrescription p;
        std::string e;
        loadBuiltinLens(kv.first, p, &e);
        RealLens raw;
        RealLensSettings s;
        s.bestFocus = false;
        raw.setup(p, s, &e);
        s.focalMm = std::fabs(raw.efl()) * kv.second;   // same unit as the Optiland sample
        RealLens L;
        L.setup(p, s, &e);
        lenses[kv.first] = L;
        // one pixel of a 2K image whose corner is the design field
        const double corner = std::fabs(L.fieldScale()) * std::tan(p.designHalfFieldDeg * kPi / 180.);
        pixel[kv.first] = corner / std::hypot(1024., 576.);
    }
    double worstEfl = 0., worstChief = 0., worstRay = 0.;
    int chief = 0, rays = 0, both = 0;
    for (const RefRow& r : rows) {
        RealLens& L = lenses[r.lens];
        if (r.kind == "efl") {
            worstEfl = std::max(worstEfl, std::fabs(L.eflAt(r.wl * 1000.) / r.value - 1.));
            continue;
        }
        const double Y = L.fieldScale() * std::tan(r.field * kPi / 180.);
        double y = std::nan("");
        if (r.kind == "chief") {
            double x;
            L.chiefLanding(0., Y, r.wl * 1000., x, y);
        } else {
            RealLens::Ray ray = L.rayThroughEntrancePupil(L.fieldAt(0., Y), 0., r.pupil * L.entrancePupilRadius());
            if ( L.trace(ray, r.wl * 1000., false) ) y = ray.p.y;
        }
        if ( std::isnan(y) && std::isnan(r.value) ) { ++both; continue; }   // missed by both tracers
        const double d = std::isnan(y) || std::isnan(r.value) ? 1e9 : std::fabs(y - r.value) / pixel[r.lens];
        if (r.kind == "chief") { worstChief = std::max(worstChief, d); ++chief; }
        else { worstRay = std::max(worstRay, d); ++rays; }
    }
    CHECK(worstEfl < 1e-8, "paraxial focal length: relative difference %.1e", worstEfl);
    CHECK(worstChief < 0.1, "%d chief rays (distortion, lateral colour): largest difference %.2e pixel at 2K", chief, worstChief);
    CHECK(worstRay < 0.1, "%d rays through the entrance pupil: largest difference %.2e pixel at 2K (%d missed by both)", rays, worstRay, both);
}

static void testWavefronts()
{
    std::printf("Wavefront fit (45 Zernike terms) and its consistency with the traced rays\n");
    ImageGeometry g;
    for (int i = 0; i < builtinLensCount(); ++i) {
        LensPrescription p;
        loadBuiltinLens(builtinLensId(i), p);
        RealLensSettings s;
        s.focalMm = 50.;
        RealLens L;
        std::string e;
        L.setup(p, s, &e);
        // a sensor whose corner is the design field
        g.filmbackWidthMm = 2. * 50. * std::tan(p.designHalfFieldDeg * kPi / 180.) * 2048. / std::hypot(2048., 1152.);
        RealLensModel M(L);
        std::vector<double> nm = { 587.5618 };
        M.prepare(g, nm, std::function<void(int, int)>(), &e);
        double worstGrad = 0., worstFit = 0., rmsAxis = 0., rmsCorner = 0.;
        int ok = 0;
        for (double fr : { 0., 0.5, 0.7, 0.9 }) {
            const RealLensModel::FieldWavefront w = M.fieldWavefront(fr * M.maxFieldMm(), 587.5618, false);
            if (!w.ok) continue;
            ++ok;
            if (w.rmsUm > 0.01) worstGrad = std::max(worstGrad, std::fabs(w.gradientCheck - 1.));
            worstFit = std::max(worstFit, w.fitResidualUm / std::max(w.rmsUm, 0.05));
            if (fr == 0.) rmsAxis = w.rmsUm / 0.5876;
            rmsCorner = w.rmsUm / 0.5876;
        }
        CHECK(ok == 4 && worstGrad < 0.03 && worstFit < 0.05,
              "%-17s f/%.1f: RMS wavefront %.2f waves on axis, %.2f at 0.9 field; -2N grad W predicts the rays within %.1f %%, fit residual %.1f %%",
              builtinLensId(i), L.fNumber(), rmsAxis, rmsCorner, 100. * worstGrad, 100. * worstFit);
    }
}

static void testRender()
{
    std::printf("Real lens in the renderer (double Gauss 50 mm f/2, 7 blades; image corner at 95 %% of the design field)\n");
    LensPrescription p;
    loadBuiltinLens("double_gauss_f2", p);
    RealLensSettings s;
    s.focalMm = 50.;
    s.fNumber = 2.;
    s.blades = 7;
    RealLens L;
    std::string e;
    L.setup(p, s, &e);
    RealLensModel model(L);
    ImageGeometry g;
    g.width = 512;
    g.height = 288;
    const double corner = 0.95 * 50. * std::tan(p.designHalfFieldDeg * kPi / 180.);
    g.filmbackWidthMm = 2. * corner * 512. / std::hypot(512., 288.);
    RenderSettings rs;
    rs.kernelSize = 33;
    rs.wavelengths = 8;
    LensPlan plan;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = buildLensPlan(model, g, rs, plan, std::function<void(int, int)>(), &e);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ok, "lens plan built in %.2f s (%s)", secs, ok ? "ok" : e.c_str());
    if (!ok) return;
    // displacement in the corner = chief ray landing - ideal position
    const int l = plan.wavelengthCount() / 2;
    double hx, hy;
    pixelToField(g, 0.5, 0.5, hx, hy);
    double dx, dy;
    model.displacementUm(hx, hy, l, dx, dy);
    const double X = hx * model.maxFieldMm(), Y = hy * model.maxFieldMm();
    double lx, ly;
    L.chiefLanding(X, Y, plan.spectral.wavelengths[l].nm, lx, ly);
    const double err = std::hypot(dx * 1e-3 - (lx - X), dy * 1e-3 - (ly - Y)) / (g.pixelPitchUm() * 1e-3);
    CHECK(err < 0.05, "corner displacement %.3f px, chief ray says %.3f px (difference %.3f px)",
          std::hypot(dx, dy) / g.pixelPitchUm(), std::hypot(lx - X, ly - Y) / (g.pixelPitchUm() * 1e-3), err);
    // illumination: 1 on axis, falls towards the corner
    const float c = plan.illum[(size_t)(plan.mapH / 2) * plan.mapW + plan.mapW / 2];
    const float k = model.illumination(hx, hy);
    CHECK(std::fabs(c - 1.f) < 0.02f && k < c && k > 0.2f, "relative illumination: centre %.3f, corner %.3f", c, k);
    // a point light keeps its light, minus the vignetting at that place
    ImageRGB in, out;
    in.resize(g.width, g.height);
    const int px = 60, py = 50;
    for (int ch = 0; ch < 3; ++ch) in.c[ch][(size_t)py * g.width + px] = 1.f;
    renderCpu(plan, in, out);
    double sum = 0.;
    for (float v : out.c[1]) sum += v;
    double fx, fy;
    pixelToField(g, px + 0.5, py + 0.5, fx, fy);
    const double expect = model.illumination(fx, fy);
    CHECK(std::fabs(sum / expect - 1.) < 0.03, "a point light off axis keeps %.3f of its light (relative illumination there: %.3f)", sum, expect);
    // PSF on fine pixels (1 um): sharp on axis, larger towards the corner; diffraction stars
    SensorParams sp;
    sp.pixelPitchUm = 1.;
    sp.kernelSize = 121;
    double ee[2];
    const double fields[2] = { 0., 0.9 };
    for (int i = 0; i < 2; ++i) {
        WavefrontParams w;
        ApertureParams a;
        model.psfInputs(0., fields[i], l, w, a);
        PsfResult r = computePsf(w, a, sp, 0., fields[i], plan.spectral.wavelengths[l].nm);
        ee[i] = r.ee80Px;
    }
    const double airy = 1.2197 * plan.spectral.wavelengths[l].nm * 1e-3 * 2.;
    CHECK(ee[0] > 0. && ee[0] < 3. * airy && ee[1] > ee[0], "80 %% of the light within %.2f um on axis (Airy radius %.2f um), %.2f um at 0.9 field",
          ee[0], airy, ee[1]);
}

// Step 5: wavelengths between the traced ones are interpolated (Lagrange in 1/lambda^2);
// automatic PSF size and automatic Final grid
static void testFinalMode()
{
    std::printf("Final mode: wavelength interpolation, automatic PSF size and grid (double Gauss 50 mm f/2)\n");
    LensPrescription p;
    loadBuiltinLens("double_gauss_f2", p);
    RealLensSettings s;
    s.focalMm = 50.;
    s.fNumber = 2.;
    RealLens L;
    std::string e;
    L.setup(p, s, &e);
    ImageGeometry g;
    g.width = 512;
    g.height = 288;
    const double corner = 0.9 * 50. * std::tan(p.designHalfFieldDeg * kPi / 180.);
    g.filmbackWidthMm = 2. * corner * 512. / std::hypot(512., 288.);
    std::vector<double> nm;
    for (int i = 0; i < 24; ++i) nm.push_back(400. + 300. * i / 23.);
    RealLensModel all(L), interp(L);
    all.setMaxTracedWavelengths(24);
    interp.setMaxTracedWavelengths(9);
    all.prepare(g, nm, std::function<void(int, int)>(), &e);
    interp.prepare(g, nm, std::function<void(int, int)>(), &e);
    double worstDisp = 0., worstPsf = 0.;
    for (int l = 0; l < 24; ++l) {
        for (double h : { 0., 0.6, 0.95 }) {
            const double hx = 0.8 * h, hy = 0.6 * h;
            double ax, ay, bx, by;
            all.displacementUm(hx, hy, l, ax, ay);
            interp.displacementUm(hx, hy, l, bx, by);
            worstDisp = std::max(worstDisp, std::hypot(ax - bx, ay - by));
        }
        // PSF at the corner, L1 difference
        const double px = 0.95 * g.width, py = 0.95 * g.height;
        PsfOptions o;
        const PsfResult a = computePsfAt(all, g, 33, px, py, l, nm[l], o);
        const PsfResult b = computePsfAt(interp, g, 33, px, py, l, nm[l], o);
        double d = 0., n = 0.;
        for (size_t k = 0; k < a.kernel.size(); ++k) {
            d += std::fabs(a.kernel[k] - b.kernel[k]);
            n += a.kernel[k];
        }
        worstPsf = std::max(worstPsf, d / std::max(n, 1e-12));
    }
    CHECK(interp.tracedWavelengths() == 9 && worstDisp < 0.3 && worstPsf < 0.02,
          "24 wavelengths from 9 traced ones: chief ray within %.3f um, corner PSF within %.2f %% (L1)",
          worstDisp, 100. * worstPsf);

    // automatic PSF size: smaller than the cap for this sharp lens, and the PSFs still hold their light
    RenderSettings rs;
    rs.kernelSize = 129;
    rs.wavelengths = 8;
    rs.autoKernel = true;
    LensPlan plan;
    bool ok = buildLensPlan(interp, g, rs, plan, std::function<void(int, int)>(), &e);
    CHECK(ok && plan.S < 129 && plan.minEnergyInKernel > 0.97,
          "automatic PSF size %d px (cap 129), at least %.1f %% of each PSF's light inside", plan.S, 100. * plan.minEnergyInKernel);

    // automatic grid: refined until the measured interpolation error meets the tolerance
    rs.autoKernel = true;
    rs.autoGrid = true;
    rs.gridTolerance = 0.02;
    rs.gridX = 3;
    rs.gridY = 3;
    LensPlan fine;
    ok = buildLensPlan(interp, g, rs, fine, std::function<void(int, int)>(), &e);
    CHECK(ok && fine.gridError >= 0. && fine.gridError <= 0.02 && fine.nodeCount() > 9 && fine.gridX <= rs.gridMax,
          "automatic grid: 3 x 3 refined to %d x %d, interpolation error %.2f %% (tolerance 2 %%)",
          fine.gridX, fine.gridY, 100. * fine.gridError);
}

int main(int argc, char** argv)
{
    if (argc > 1) gRef = argv[1];
    testGlasses();
    testRayTracing();
    testWavefronts();
    testRender();
    testFinalMode();
    std::printf("\n%d checks, %d failures\n", gChecks, gFail);
    return gFail ? 1 : 0;
}
