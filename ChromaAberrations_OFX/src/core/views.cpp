// ChromaAberrations_OFX - test chart and PSF map views (plugin and tools)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "views.h"

#include <algorithm>
#include <cmath>

namespace ca {

void makeTestChart(const std::string& kind, int w, int h, ImageRGB& img)
{
    img.resize(w, h);
    const bool siemens = (kind == "siemens") || (kind == "chart");
    const bool points = (kind == "points") || (kind == "chart");
    if (siemens) {
        // Siemens stars (36 spokes) at the centre and towards the corners, on mid grey
        const int nx = 5, ny = 3;
        const double rad = 0.42 * std::min( (double)w / nx, (double)h / ny );
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const double cx = (std::floor( (x + 0.5) * nx / w ) + 0.5) * w / nx;
                const double cy = (std::floor( (y + 0.5) * ny / h ) + 0.5) * h / ny;
                const double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                const double r = std::hypot(dx, dy);
                float v = 0.18f;
                if (r < rad) {
                    int on = 0;   // 4 x 4 supersampled spokes
                    for (int j = 0; j < 4; ++j) {
                        for (int i = 0; i < 4; ++i) {
                            const double t = std::atan2(dy - 0.5 + (j + 0.5) / 4., dx - 0.5 + (i + 0.5) / 4.);
                            on += std::sin(18. * t) > 0.;
                        }
                    }
                    v = 0.02f + 0.78f * on / 16.f;
                }
                for (int c = 0; c < 3; ++c) {
                    img.c[c][(size_t)y * w + x] = v;
                }
            }
        }
    }
    if (points) {
        // very bright point lights (HDR), white and a few saturated colours
        const int step = std::max(16, std::min(w, h) / 9);
        int n = 0;
        for (int y = step / 2; y < h; y += step) {
            for (int x = step / 2; x < w; x += step, ++n) {
                float c[3] = { 1.f, 1.f, 1.f };
                if (n % 7 == 3) { c[0] = 1.f; c[1] = 0.05f; c[2] = 0.02f; }
                if (n % 7 == 5) { c[0] = 0.05f; c[1] = 0.25f; c[2] = 1.f; }
                for (int k = 0; k < 3; ++k) {
                    img.c[k][(size_t)y * w + x] += 200.f * c[k];
                }
            }
        }
    }
}

void makePsfMapView(const LensPlan& plan, const ImageRGB* background, int zoom, ImageRGB& out)
{
    const int W = plan.geom.width, H = plan.geom.height;
    out.resize(W, H);
    if ( background && (background->width == W) && (background->height == H) ) {
        for (int c = 0; c < 3; ++c) {
            for (size_t i = 0; i < out.c[c].size(); ++i) {
                out.c[c][i] = 0.05f * background->c[c][i];
            }
        }
    }
    const int S = plan.S, P = S * S, nl = plan.wavelengthCount(), G = plan.nodeCount();
    // white-light PSF of every grid point
    std::vector<std::vector<float> > white(G);
    double extent = 0.5;   // half-size (pixels) of the square holding 95 % of the light, largest over the grid
    for (int g = 0; g < G; ++g) {
        std::vector<float>& rgb = white[g];
        rgb.assign( (size_t)3 * P, 0.f );
        for (int l = 0; l < nl; ++l) {
            const float* k = &plan.bankKernels[l][(size_t)g * P];
            for (int c = 0; c < 3; ++c) {
                const float wc = (float)plan.spectral.response[c][l];
                for (int p = 0; p < P; ++p) {
                    rgb[(size_t)c * P + p] += wc * k[p];
                }
            }
        }
        const int R = S / 2;
        double total = 0.;
        for (int p = 0; p < P; ++p) total += std::fabs(rgb[(size_t)P + p]);
        for (int h = 0; h <= R; ++h) {
            double in = 0.;
            for (int y = R - h; y <= R + h; ++y) for (int x = R - h; x <= R + h; ++x) in += std::fabs(rgb[(size_t)P + (size_t)y * S + x]);
            if (in >= 0.95 * total) {
                extent = std::max(extent, h + 0.5);
                break;
            }
        }
    }
    if (zoom <= 0) {
        // automatic: the largest PSF fills 80 % of a grid cell
        const double cell = std::min( plan.gridX > 1 ? (double)W / (plan.gridX - 1) : (double)W,
                                      plan.gridY > 1 ? (double)H / (plan.gridY - 1) : (double)H );
        zoom = std::max( 1, std::min( 64, (int)(0.8 * cell / (2. * extent)) ) );
    }
    // only the useful part of the kernel is drawn
    const int half = std::min( S / 2, (int)std::ceil(extent) );
    const int side = (2 * half + 1) * zoom;
    for (int gy = 0; gy < plan.gridY; ++gy) {
        for (int gx = 0; gx < plan.gridX; ++gx) {
            const int g = gy * plan.gridX + gx;
            const std::vector<float>& rgb = white[g];
            float peak = 0.f;
            for (int p = 0; p < P; ++p) {
                peak = std::max(peak, rgb[(size_t)1 * P + p]);
            }
            const float norm = peak > 0.f ? 1.f / peak : 0.f;
            // centre of the patch at the grid point, kept inside the image
            const int cx = std::min( std::max( (int)plan.nodeX(gx), side / 2 ), std::max(side / 2, W - 1 - side / 2) );
            const int cy = std::min( std::max( (int)plan.nodeY(gy), side / 2 ), std::max(side / 2, H - 1 - side / 2) );
            for (int y = 0; y < side; ++y) {
                const int oy = cy - side / 2 + y;
                if ( (oy < 0) || (oy >= H) ) {
                    continue;
                }
                for (int x = 0; x < side; ++x) {
                    const int ox = cx - side / 2 + x;
                    if ( (ox < 0) || (ox >= W) ) {
                        continue;
                    }
                    const size_t kp = (size_t)(S / 2 - half + y / zoom) * S + (S / 2 - half + x / zoom);
                    const size_t o = (size_t)oy * W + ox;
                    for (int c = 0; c < 3; ++c) {
                        // display curve (gamma 2.2) so that the faint parts of the PSF show
                        const float v = std::max(0.f, rgb[(size_t)c * P + kp] * norm);
                        out.c[c][o] = std::max(out.c[c][o], std::pow(v, 1.f / 2.2f));
                    }
                }
            }
        }
    }
}

} // namespace ca
