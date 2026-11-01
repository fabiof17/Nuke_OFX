// ChromaAberrations_OFX - aperture (diaphragm) shape
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "aperture.h"

#include <algorithm>
#include <cmath>

namespace ca {

static const double kPi = 3.14159265358979323846;

void vignettingCircle(const ApertureParams& a, double& radius, double& shiftPerField)
{
    // the circle edge reaches the pupil edge at |H| = start, and cuts 'vignetting' deep at |H| = 1
    const double start = std::min( std::max(a.vignettingStart, 0.), 0.95 );
    shiftPerField = std::max(a.vignetting, 0.) / (1. - start);
    radius = 1. + shiftPerField * start;
}

bool apertureInside(const ApertureParams& a, double x, double y)
{
    if (a.vignetting > 0.) {
        double rv, k;
        vignettingCircle(a, rv, k);
        const double vx = x - k * a.fieldX, vy = y - k * a.fieldY;
        if (vx * vx + vy * vy > rv * rv) {
            return false;
        }
    }
    // oval: the longer axis keeps radius 1
    if (a.aspect > 1.) {
        x *= a.aspect;
    } else if (a.aspect < 1. && a.aspect > 0.) {
        y /= a.aspect;
    }
    if (a.mask && (a.maskN > 1)) {
        const int n = a.maskN;
        const double u = (x + 1.) * 0.5 * (n - 1), v = (y + 1.) * 0.5 * (n - 1);
        if ( (u < 0.) || (v < 0.) || (u > n - 1) || (v > n - 1) ) {
            return false;
        }
        const int i = std::min( (int)u, n - 2 ), j = std::min( (int)v, n - 2 );
        const double fu = u - i, fv = v - j;
        const std::vector<float>& m = *a.mask;
        const double t = (1. - fv) * ( (1. - fu) * m[(size_t)j * n + i] + fu * m[(size_t)j * n + i + 1] ) +
                         fv * ( (1. - fu) * m[(size_t)(j + 1) * n + i] + fu * m[(size_t)(j + 1) * n + i + 1] );
        if (t < 0.5) {
            return false;
        }
    }
    if (a.hasPupilMap) {
        const double px = a.pupilMap[0] * x + a.pupilMap[1] * y, py = a.pupilMap[2] * x + a.pupilMap[3] * y;
        x = px;
        y = py;
    }
    const double r = std::sqrt(x * x + y * y);
    if (r > 1.) {
        return false;
    }
    if (a.blades < 3 || a.roundness >= 1.) {
        return true;
    }
    // regular polygon with its vertices on the unit circle
    const double seg = 2. * kPi / a.blades;
    double t = std::atan2(y, x) - a.rotationDeg * kPi / 180.;
    t = std::fmod(t, seg);
    if (t < 0.) {
        t += seg;
    }
    const double rPoly = std::cos(kPi / a.blades) / std::cos(t - kPi / a.blades);
    const double rMax = rPoly + (1. - rPoly) * std::max(0., a.roundness);
    return r <= rMax;
}

double apertureCoverage(const ApertureParams& a, double x, double y, double cell)
{
    // fully inside or outside when the 4 corners agree and the cell is far from the rim
    const double h = 0.5 * cell;
    const bool c0 = apertureInside(a, x - h, y - h), c1 = apertureInside(a, x + h, y - h);
    const bool c2 = apertureInside(a, x - h, y + h), c3 = apertureInside(a, x + h, y + h);
    if (c0 == c1 && c1 == c2 && c2 == c3) {
        // a blade edge could still cross the cell between corners only if the cell is
        // larger than the polygon's features, which never happens at the sizes used
        return c0 ? 1. : 0.;
    }
    int n = 0;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            if ( apertureInside(a, x - h + (i + 0.5) * cell / 4., y - h + (j + 0.5) * cell / 4.) ) {
                ++n;
            }
        }
    }
    return n / 16.;
}

double apertureTransmission(const ApertureParams& a, int samples)
{
    if (a.vignetting <= 0.) {
        return 1.;
    }
    ApertureParams axis = a;
    axis.fieldX = axis.fieldY = 0.;
    long long open = 0, full = 0;
    for (int j = 0; j < samples; ++j) {
        for (int i = 0; i < samples; ++i) {
            const double x = -1. + (i + 0.5) * 2. / samples, y = -1. + (j + 0.5) * 2. / samples;
            if ( apertureInside(axis, x, y) ) {
                ++full;
                open += apertureInside(a, x, y);
            }
        }
    }
    return full ? (double)open / full : 1.;
}

} // namespace ca
