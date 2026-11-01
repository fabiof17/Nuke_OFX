// ChromaAberrations_OFX - everything the renderer needs from the lens, computed once per setting
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "lensplan.h"
#include "lensmodel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

namespace ca {

void LensPlan::gridCoords(double x, double y, double& u, double& v) const
{
    u = gridX > 1 ? x * (gridX - 1) / geom.width : 0.;
    v = gridY > 1 ? y * (gridY - 1) / geom.height : 0.;
    u = std::min( std::max(u, 0.), (double)(gridX - 1) );
    v = std::min( std::max(v, 0.), (double)(gridY - 1) );
}

LensPlan buildLensPlan(const WavefrontParams& w, const ApertureParams& a, const ImageGeometry& g,
                       const RenderSettings& rs, std::function<void(int, int)> progress)
{
    ParametricLensModel model(w, a);
    std::string error;
    LensPlan plan;
    buildLensPlan(model, g, rs, plan, progress, &error);
    return plan;
}

bool buildLensPlan(LensModel& model, const ImageGeometry& g, const RenderSettings& rs, LensPlan& plan,
                   std::function<void(int, int)> progress, std::string* error)
{
    const auto t0 = std::chrono::steady_clock::now();
    plan = LensPlan();
    plan.geom = g;
    plan.settings = rs;
    plan.spectral = buildSpectralBasis(rs.wavelengths, rs.colorSpace, 400., 700., rs.camera.get());
    const int nl = plan.spectral.count();
    const double pitch = g.pixelPitchUm();
    {
        std::vector<double> nm;
        for (const WavelengthSample& s : plan.spectral.wavelengths) {
            nm.push_back(s.nm);
        }
        if ( !model.prepare(g, nm, progress, error) ) {
            return false;
        }
    }
    auto tPrepared = std::chrono::steady_clock::now();
    plan.prepareSeconds = std::chrono::duration<double>(tPrepared - t0).count();

    // ---- PSFs
    if (rs.blur) {
        plan.S = std::max(3, rs.kernelSize | 1);
        plan.gridX = std::max(1, rs.gridX);
        plan.gridY = std::max(1, rs.gridY);
        int midL = 0;
        for (int l = 0; l < nl; ++l) {
            if (std::fabs(plan.spectral.wavelengths[l].nm - 550.) < std::fabs(plan.spectral.wavelengths[midL].nm - 550.)) midL = l;
        }
        // automatic PSF size: the largest PSF over the grid and the spectrum, up to kernelSize
        if (rs.autoKernel) {
            double maxR = 0.;
            const int ls[3] = { 0, midL, nl - 1 };
            for (int gy = 0; gy < plan.gridY; ++gy) {
                for (int gx = 0; gx < plan.gridX; ++gx) {
                    const double px = plan.gridX > 1 ? g.width * gx / (plan.gridX - 1.) : 0.5 * g.width;
                    const double py = plan.gridY > 1 ? g.height * gy / (plan.gridY - 1.) : 0.5 * g.height;
                    for (int l : ls) {
                        maxR = std::max( maxR, psfExtentPx(model, g, px, py, l, plan.spectral.wavelengths[l].nm) );
                    }
                }
            }
            const int need = 2 * (int)std::ceil(maxR + 1.) + 1;
            plan.S = std::min( plan.S, std::max(9, need | 1) );
        }
        // automatic grid: refine until the PSF interpolated at the middle of the cells (the worst
        // place for bilinear interpolation) is within the tolerance of the PSF computed there
        if (rs.autoGrid) {
            const double wl = plan.spectral.wavelengths[midL].nm;
            for (int iter = 0; iter < 5; ++iter) {
                const int gx = plan.gridX, gy = plan.gridY;
                std::vector<std::pair<double, double> > pos;
                for (int j = 0; j < gy; ++j) for (int i = 0; i < gx; ++i) {
                    pos.push_back( std::make_pair( gx > 1 ? g.width * i / (gx - 1.) : 0.5 * g.width,
                                                   gy > 1 ? g.height * j / (gy - 1.) : 0.5 * g.height ) );
                }
                const int nNodes = (int)pos.size();
                for (int j = 0; j + 1 < gy; ++j) for (int i = 0; i + 1 < gx; ++i) {
                    pos.push_back( std::make_pair( g.width * (i + 0.5) / (gx - 1.), g.height * (j + 0.5) / (gy - 1.) ) );
                }
                std::vector<std::vector<float> > k(pos.size());
                std::atomic<int> next(0);
                auto work = [&]() {
                    for (;;) {
                        const int t = next++;
                        if (t >= (int)pos.size()) return;
                        PsfResult r = computePsfAt(model, g, plan.S, pos[t].first, pos[t].second, midL, wl, rs.psf);
                        double s = 0.;
                        for (float v : r.kernel) s += v;
                        k[t] = r.kernel;
                        if (s > 0.) for (float& v : k[t]) v = (float)(v / s);
                    }
                };
                std::vector<std::thread> th;
                const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
                for (unsigned t = 0; t < nt; ++t) th.emplace_back(work);
                for (std::thread& t : th) t.join();
                // error compared after a slight blur (sigma 0.7 px): sub-pixel differences between
                // tiny PSFs do not show in the image and must not drive the grid
                const int S = plan.S;
                auto soften = [&](std::vector<float> v) {
                    const double sig = 0.7;
                    const float w[3] = { (float)std::exp(-1. / (2. * sig * sig)), 1.f, (float)std::exp(-1. / (2. * sig * sig)) };
                    const float ws = w[0] + w[1] + w[2];
                    std::vector<float> t(v.size(), 0.f);
                    for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
                        float a = 0.f;
                        for (int d = -1; d <= 1; ++d) if (x + d >= 0 && x + d < S) a += w[d + 1] * v[(size_t)y * S + x + d];
                        t[(size_t)y * S + x] = a / ws;
                    }
                    for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
                        float a = 0.f;
                        for (int d = -1; d <= 1; ++d) if (y + d >= 0 && y + d < S) a += w[d + 1] * t[(size_t)(y + d) * S + x];
                        v[(size_t)y * S + x] = a / ws;
                    }
                    return v;
                };
                double err = 0.;
                int c = nNodes;
                for (int j = 0; j + 1 < gy; ++j) for (int i = 0; i + 1 < gx; ++i, ++c) {
                    const std::vector<float>& a = k[j * gx + i];
                    const std::vector<float>& b = k[j * gx + i + 1];
                    const std::vector<float>& d = k[(j + 1) * gx + i];
                    const std::vector<float>& e = k[(j + 1) * gx + i + 1];
                    std::vector<float> interp(k[c].size());
                    for (size_t p = 0; p < interp.size(); ++p) interp[p] = 0.25f * (a[p] + b[p] + d[p] + e[p]);
                    const std::vector<float> tc = soften(k[c]), ti = soften(interp);
                    double num = 0., den = 0.;
                    for (size_t p = 0; p < tc.size(); ++p) {
                        num += std::fabs(tc[p] - ti[p]);
                        den += std::fabs(tc[p]);
                    }
                    if (den > 0.) err = std::max(err, num / den);
                }
                plan.gridError = err;
                if ( (gx < 2) || (gy < 2) || (err <= rs.gridTolerance) ) break;
                // bilinear interpolation error goes as the square of the cell size
                const double f = std::min( 3., std::sqrt(err / std::max(1e-6, rs.gridTolerance)) * 1.1 );
                int nx = std::min( rs.gridMax, (int)std::ceil( (gx - 1) * f ) + 1 );
                int ny = std::min( rs.gridMax, (int)std::ceil( (gy - 1) * f ) + 1 );
                // budget: at most gridMax x gridMax * 0.6 grid points
                while ( (nx * ny > rs.gridMax * rs.gridMax * 6 / 10) && (nx > 2) && (ny > 2) ) {
                    nx = std::max(2, nx - 1);
                    ny = std::max(2, (int)std::lround( (double)ny * (nx - 1.) / std::max(1., nx * 1.) ) + 1);
                }
                if ( (nx == gx) && (ny == gy) ) break;
                plan.gridX = nx;
                plan.gridY = ny;
            }
        }
        plan.R = plan.S / 2;
        PsfBank bank = computePsfBank(model, g, plan.S, plan.gridX, plan.gridY, plan.spectral.wavelengths, rs.psf, progress);
        const int G = plan.gridX * plan.gridY, P = plan.S * plan.S;
        plan.bankKernels.resize(nl);
        for (int l = 0; l < nl; ++l) {
            std::vector<float>& bk = plan.bankKernels[l];
            bk.assign( (size_t)G * P, 0.f );
            for (int cell = 0; cell < G; ++cell) {
                const PsfResult& r = bank.psf[(size_t)cell * nl + l];
                double s = 0.;
                for (float v : r.kernel) {
                    s += v;
                }
                plan.minEnergyInKernel = std::min(plan.minEnergyInKernel, r.energyInKernel);
                const double k = s > 0. ? 1. / s : 0.;
                for (int p = 0; p < P; ++p) {
                    bk[(size_t)cell * P + p] = (float)(r.kernel[p] * k);
                }
            }
        }
    } else {
        // no blur: a single delta kernel
        plan.S = 1;
        plan.R = 0;
        plan.gridX = plan.gridY = 1;
        plan.bankKernels.assign( nl, std::vector<float>(1, 1.f) );
    }

    auto tPsf = std::chrono::steady_clock::now();
    plan.psfSeconds = std::chrono::duration<double>(tPsf - tPrepared).count();

    // ---- displacement and illumination maps, covering the padded image
    const double step = std::max(2, rs.mapStep);
    plan.mapStep = step;
    plan.mapX0 = -plan.R - 2. * step;
    plan.mapY0 = -plan.R - 2. * step;
    plan.mapW = (int)std::ceil( (g.width + 2. * plan.R + 4. * step) / step ) + 1;
    plan.mapH = (int)std::ceil( (g.height + 2. * plan.R + 4. * step) / step ) + 1;
    const size_t nodes = (size_t)plan.mapW * plan.mapH;
    plan.dispX.assign( nl, std::vector<float>(nodes, 0.f) );
    plan.dispY.assign( nl, std::vector<float>(nodes, 0.f) );
    plan.illum.assign(nodes, 1.f);
    const double cx = g.centerX * g.width, cy = g.centerY * g.height;
    for (int j = 0; j < plan.mapH; ++j) {
        for (int i = 0; i < plan.mapW; ++i) {
            const double px = plan.mapX0 + i * step, py = plan.mapY0 + j * step;
            double hx, hy;
            pixelToField(g, px, py, hx, hy);
            const size_t n = (size_t)j * plan.mapW + i;
            if ( rs.vignetting && !model.includesNaturalFalloff() ) {
                const double rmm = std::hypot( (px - cx) * g.pixelAspect, py - cy ) * pitch * 1e-3;
                const double c2 = 1. / (1. + std::pow(rmm / std::max(rs.focalLengthMm, 1e-3), 2.));   // cos^2
                plan.illum[n] = (float)(1. - rs.naturalVignetting + rs.naturalVignetting * c2 * c2);
            }
        }
    }
    if (rs.distortion) {
        // The renderer needs, for each landing position x, the image point x_s whose chief ray
        // lands there: x_s + D(x_s) = x, with D the displacement of the model. Solved by Newton
        // in double precision for every map node (strong barrel distortion included), stored as
        // the inverse displacement x - x_s.
        auto D = [&](int l, double px, double py, double& dx, double& dy) {
            double hx, hy;
            pixelToField(g, px, py, hx, hy);
            model.displacementUm(hx, hy, l, dx, dy);
            dx /= pitch * g.pixelAspect;   // desqueezed micrometres -> plate pixels
            dy /= pitch;
        };
        std::vector<double> maxD(nl, 0.);
        auto work = [&](int l) {
            for (int j = 0; j < plan.mapH; ++j) {
                for (int i = 0; i < plan.mapW; ++i) {
                    const double x = plan.mapX0 + i * step, y = plan.mapY0 + j * step;
                    double dx, dy;
                    D(l, x, y, dx, dy);
                    maxD[l] = std::max( maxD[l], std::hypot(dx, dy) );
                    double sx = x - dx, sy = y - dy;
                    const double h = 0.25;
                    for (int it = 0; it < 30; ++it) {
                        double fx, fy, ax, ay, bx, by;
                        D(l, sx, sy, fx, fy);
                        const double rx = sx + fx - x, ry = sy + fy - y;
                        if (std::hypot(rx, ry) < 1e-5) break;
                        D(l, sx + h, sy, ax, ay);
                        D(l, sx, sy + h, bx, by);
                        const double j00 = 1. + (ax - fx) / h, j01 = (bx - fx) / h, j10 = (ay - fy) / h, j11 = 1. + (by - fy) / h;
                        const double det = j00 * j11 - j01 * j10;
                        if (std::fabs(det) < 1e-12) break;
                        double ux = ( j11 * rx - j01 * ry) / det, uy = (-j10 * rx + j00 * ry) / det;
                        const double len = std::hypot(ux, uy), lim = 0.25 * (std::fabs(x - sx) + std::fabs(y - sy)) + 8.;
                        if (len > lim) { ux *= lim / len; uy *= lim / len; }   // damped steps far from the solution
                        sx -= ux;
                        sy -= uy;
                    }
                    const size_t n = (size_t)j * plan.mapW + i;
                    plan.dispX[l][n] = (float)(x - sx);
                    plan.dispY[l][n] = (float)(y - sy);
                }
            }
        };
        std::vector<std::thread> th;
        for (int l = 0; l < nl; ++l) th.emplace_back(work, l);
        for (std::thread& t : th) t.join();
        plan.maxDisplacementPx = *std::max_element(maxD.begin(), maxD.end());
    }
    if (rs.vignetting) {
        // illumination of the lens model (mechanical vignetting, or everything for a real lens),
        // on a coarser grid: it varies slowly
        const int sub = model.cheapIllumination() ? 1 : std::max(1, std::min(4, plan.mapW / 32));   // at least ~32 nodes across
        const int cw = (plan.mapW + sub - 1) / sub + 1, ch = (plan.mapH + sub - 1) / sub + 1;
        std::vector<double> tr( (size_t)cw * ch, 1. );
        std::vector<std::thread> th;
        const int nt = std::max(1u, std::thread::hardware_concurrency());
        for (int t = 0; t < nt; ++t) {
            th.emplace_back([&, t]() {
                for (int idx = t; idx < cw * ch; idx += nt) {
                    const int i = idx % cw, j = idx / cw;
                    double hx, hy;
                    pixelToField(g, plan.mapX0 + i * sub * step, plan.mapY0 + j * sub * step, hx, hy);
                    tr[idx] = model.illumination(hx, hy);
                }
            });
        }
        for (std::thread& t : th) {
            t.join();
        }
        for (int j = 0; j < plan.mapH; ++j) {
            for (int i = 0; i < plan.mapW; ++i) {
                const double u = (double)i / sub, v = (double)j / sub;
                const int i0 = std::min( (int)u, cw - 2 ), j0 = std::min( (int)v, ch - 2 );
                const double fu = u - i0, fv = v - j0;
                const double t = (1. - fv) * ( (1. - fu) * tr[(size_t)j0 * cw + i0] + fu * tr[(size_t)j0 * cw + i0 + 1] ) +
                                 fv * ( (1. - fu) * tr[(size_t)(j0 + 1) * cw + i0] + fu * tr[(size_t)(j0 + 1) * cw + i0 + 1] );
                float& il = plan.illum[(size_t)j * plan.mapW + i];
                if ( model.includesNaturalFalloff() ) {
                    il = (float)(1. - rs.naturalVignetting + rs.naturalVignetting * t);
                } else {
                    il *= (float)t;
                }
            }
        }
    }
    plan.buildSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    plan.mapsSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tPsf).count();
    return true;
}

} // namespace ca
