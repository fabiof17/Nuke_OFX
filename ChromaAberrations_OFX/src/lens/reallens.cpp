// ChromaAberrations_OFX - real lenses: sequential ray tracing through an optical prescription
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "reallens.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

namespace ca {

static const double kPi = 3.14159265358979323846;
static const double kRefNm = 587.5618;   // helium d line: reference wavelength of the paraxial data

static inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 add(const Vec3& a, const Vec3& b, double t) { return Vec3(a.x + t * b.x, a.y + t * b.y, a.z + t * b.z); }
static inline Vec3 sub(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 normalize(const Vec3& a)
{
    const double l = std::sqrt( dot(a, a) );
    return Vec3(a.x / l, a.y / l, a.z / l);
}

// small dense solver (Gaussian elimination, partial pivoting); A is n x n row-major
static bool solve(std::vector<double>& A, std::vector<double>& b, int n)
{
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(A[(size_t)i * n + k]) > std::fabs(A[(size_t)piv * n + k])) piv = i;
        }
        if (std::fabs(A[(size_t)piv * n + k]) < 1e-300) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[(size_t)k * n + j], A[(size_t)piv * n + j]);
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = A[(size_t)i * n + k] / A[(size_t)k * n + k];
            if (f == 0.) continue;
            for (int j = k; j < n; ++j) A[(size_t)i * n + j] -= f * A[(size_t)k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[(size_t)i * n + j] * b[j];
        b[i] = s / A[(size_t)i * n + i];
    }
    return true;
}

// ============================================================================ RealLens

double RealLens::index(int i, double nm) const
{
    if (i < 0) {
        return 1.;
    }
    const Glass& g = _lens.surfaces[i].glass;
    return g.isAir() ? 1. : g.index(nm);
}

double RealLens::sag(int i, double r) const
{
    const LensSurface& s = _lens.surfaces[i];
    const double c = _c[i];
    double z = 0.;
    if (c != 0.) {
        const double q = 1. - (1. + s.conic) * c * c * r * r;
        if (q < 0.) {
            return std::nan("");
        }
        z = c * r * r / (1. + std::sqrt(q));
    }
    double rp = r * r * r * r;
    for (size_t k = 0; k < s.asph.size(); ++k, rp *= r * r) {
        z += s.asph[k] * rp;
    }
    return z;
}

double RealLens::sagSlope(int i, double r) const
{
    const LensSurface& s = _lens.surfaces[i];
    const double c = _c[i];
    double m = 0.;
    if (c != 0.) {
        const double q = 1. - (1. + s.conic) * c * c * r * r;
        if (q <= 0.) {
            return std::nan("");
        }
        m = c * r / std::sqrt(q);
    }
    double rp = r * r * r;
    for (size_t k = 0; k < s.asph.size(); ++k, rp *= r * r) {
        m += (4. + 2. * k) * s.asph[k] * rp;
    }
    return m;
}

void RealLens::normalAt(int i, double x, double y, Vec3& n) const
{
    const double r = std::sqrt(x * x + y * y);
    if (r < 1e-300) {
        n = Vec3(0., 0., 1.);
        return;
    }
    const double m = sagSlope(i, r);
    n = normalize( Vec3(-m * x / r, -m * y / r, 1.) );
}

bool RealLens::intersect(int i, Ray& ray) const
{
    const LensSurface& s = _lens.surfaces[i];
    const double zi = _z[i], c = _c[i];
    double t;
    if ( (c == 0.) && s.asph.empty() ) {
        if (std::fabs(ray.d.z) < 1e-300) return false;
        t = (zi - ray.p.z) / ray.d.z;
    } else if ( (s.conic == 0.) && s.asph.empty() ) {
        // sphere: centre on the axis at zi + R; the intersection nearest to the vertex
        const Vec3 C(0., 0., zi + 1. / c);
        const Vec3 oc = sub(ray.p, C);
        const double b = dot(ray.d, oc), cc = dot(oc, oc) - 1. / (c * c);
        const double disc = b * b - cc;
        if (disc < 0.) return false;
        const double sq = std::sqrt(disc);
        const double t1 = -b - sq, t2 = -b + sq;
        const double z1 = ray.p.z + t1 * ray.d.z, z2 = ray.p.z + t2 * ray.d.z;
        t = std::fabs(z1 - zi) < std::fabs(z2 - zi) ? t1 : t2;
    } else {
        // conic / aspheric: Newton on z(t) - zi - sag(r(t))
        t = (zi - ray.p.z) / ray.d.z;
        bool ok = false;
        for (int it = 0; it < 40; ++it) {
            const Vec3 X = add(ray.p, ray.d, t);
            const double r = std::sqrt(X.x * X.x + X.y * X.y);
            const double sg = sag(i, r), sl = sagSlope(i, r);
            if ( std::isnan(sg) || std::isnan(sl) ) return false;
            const double f = X.z - zi - sg;
            const double fp = ray.d.z - (r > 0. ? sl * (X.x * ray.d.x + X.y * ray.d.y) / r : 0.);
            if (std::fabs(fp) < 1e-300) return false;
            const double dt = f / fp;
            t -= dt;
            if (std::fabs(dt) < 1e-13 * (1. + std::fabs(t))) {
                ok = true;
                break;
            }
        }
        if (!ok) return false;
    }
    ray.p = add(ray.p, ray.d, t);
    return true;
}

bool RealLens::trace(Ray& ray, double nm, bool clip, int upTo, int* blockedAt) const
{
    const int n = (int)_lens.surfaces.size();
    const int last = (upTo < 0) ? n - 1 : std::min(upTo, n - 1);
    if (blockedAt) *blockedAt = -1;
    double n1 = 1.;
    for (int i = 0; i <= last; ++i) {
        const Vec3 p0 = ray.p;
        if ( !intersect(i, ray) ) return false;
        const Vec3 dp = sub(ray.p, p0);
        ray.opl += n1 * ( dot(dp, ray.d) );
        const double rho = std::sqrt(ray.p.x * ray.p.x + ray.p.y * ray.p.y);
        const bool over = (i == _lens.stopIndex) ? (rho > _stopR) : ( (_sd[i] > 0.) && (rho > _sd[i]) );
        if (over) {
            if (clip) return false;
            if ( blockedAt && (*blockedAt < 0) && (i != _lens.stopIndex) ) *blockedAt = i;
        }
        const double n2 = index(i, nm);
        if (n2 != n1) {
            Vec3 N;
            normalAt(i, ray.p.x, ray.p.y, N);
            double cosi = -dot(N, ray.d);
            if (cosi < 0.) {
                N = Vec3(-N.x, -N.y, -N.z);
                cosi = -cosi;
            }
            const double mu = n1 / n2;
            const double k = 1. - mu * mu * (1. - cosi * cosi);
            if (k < 0.) return false;   // total internal reflection
            ray.d = normalize( Vec3(mu * ray.d.x + (mu * cosi - std::sqrt(k)) * N.x,
                                    mu * ray.d.y + (mu * cosi - std::sqrt(k)) * N.y,
                                    mu * ray.d.z + (mu * cosi - std::sqrt(k)) * N.z) );
        }
        n1 = n2;
    }
    if (upTo < 0) {
        if (std::fabs(ray.d.z) < 1e-300) return false;
        const double t = (_zImage - ray.p.z) / ray.d.z;
        ray.p = add(ray.p, ray.d, t);
        ray.opl += n1 * t;
    }
    return true;
}

bool RealLens::paraxial(double nm, double y0, double u0, double z0, std::vector<double>& y, std::vector<double>& u) const
{
    // y-nu trace: ray of height y0 and slope u0 at z = z0 (object space)
    const int n = (int)_lens.surfaces.size();
    y.assign(n, 0.);
    u.assign(n, 0.);
    double yy = y0 + u0 * (_z[0] - z0), uu = u0, n1 = 1.;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            yy += uu * (_z[i] - _z[i - 1]);
        }
        const double n2 = index(i, nm);
        uu = (n1 * uu - yy * _c[i] * (n2 - n1)) / n2;
        y[i] = yy;
        u[i] = uu;
        n1 = n2;
    }
    return true;
}

double RealLens::eflAt(double nm) const
{
    std::vector<double> y, u;
    paraxial(nm, 1., 0., 0., y, u);
    return -1. / u.back();
}

RealLens::Field RealLens::fieldAt(double X, double Y) const
{
    Field f;
    if (_infinite) {
        f.tx = X / _k;
        f.ty = Y / _k;
    } else {
        f.object = Vec3(X / _k, Y / _k, _zObject);
    }
    return f;
}

RealLens::Ray RealLens::makeRay(const Field& f, double v0, double v1) const
{
    Ray r;
    if (_infinite) {
        r.d = normalize( Vec3(f.tx, f.ty, 1.) );
        r.p = Vec3(v0, v1, _zStart);
        r.opl = dot(r.p, r.d);   // from the plane wavefront through the origin
    } else {
        r.p = f.object;
        r.d = normalize( Vec3(v0, v1, 1.) );
        r.opl = 0.;
    }
    return r;
}

void RealLens::initialGuess(const Field& f, double sx, double sy, double v[2]) const
{
    // paraxial: the entrance pupil point conjugate to the stop point
    const double ex = sx / _stopGain, ey = sy / _stopGain;
    if (_infinite) {
        v[0] = ex + f.tx * (_zStart - _epZ);
        v[1] = ey + f.ty * (_zStart - _epZ);
    } else {
        const double dz = _epZ - f.object.z;
        v[0] = (ex - f.object.x) / dz;
        v[1] = (ey - f.object.y) / dz;
    }
}

RealLens::Ray RealLens::rayThroughEntrancePupil(const Field& f, double ex, double ey) const
{
    double v[2];
    initialGuess(f, ex * _stopGain, ey * _stopGain, v);
    return makeRay(f, v[0], v[1]);
}

bool RealLens::aim(const Field& f, double sx, double sy, double nm, double v[2], bool guess) const
{
    if (!guess) {
        initialGuess(f, sx, sy, v);
    }
    const int stop = _lens.stopIndex;
    const double scale = std::max(_stopR, 1e-6 * std::fabs(_efl));
    const double h = (_infinite ? 1e-6 * std::fabs(_efl) : 1e-7);
    auto hit = [&](double a, double b, double& x, double& y) {
        Ray r = makeRay(f, a, b);
        if ( !trace(r, nm, false, stop) ) return false;
        x = r.p.x;
        y = r.p.y;
        return true;
    };
    for (int it = 0; it < 30; ++it) {
        double x, y;
        if ( !hit(v[0], v[1], x, y) ) return false;
        const double fx = x - sx, fy = y - sy;
        if (std::hypot(fx, fy) < 1e-11 * scale + 1e-14) return true;
        double x1, y1, x2, y2;
        if ( !hit(v[0] + h, v[1], x1, y1) || !hit(v[0], v[1] + h, x2, y2) ) return false;
        const double j00 = (x1 - x) / h, j10 = (y1 - y) / h, j01 = (x2 - x) / h, j11 = (y2 - y) / h;
        const double det = j00 * j11 - j01 * j10;
        if (std::fabs(det) < 1e-300) return false;
        v[0] -= ( j11 * fx - j01 * fy) / det;
        v[1] -= (-j10 * fx + j00 * fy) / det;
    }
    return false;
}

bool RealLens::chiefLanding(double X, double Y, double nm, double& x, double& y) const
{
    const Field f = fieldAt(X, Y);
    double v[2];
    if ( !aim(f, 0., 0., nm, v) ) return false;
    Ray r = makeRay(f, v[0], v[1]);
    if ( !trace(r, nm, false) ) return false;
    x = r.p.x;
    y = r.p.y;
    return true;
}

double RealLens::onAxisRmsOpd(double zImage) const
{
    // rings of equal area along +y; OPD to the sphere centred on the axial image point
    const int n = (int)_lens.surfaces.size();
    const Field f0 = fieldAt(0., 0.);
    const Vec3 Q(0., 0., zImage);
    double vc[2];
    if ( !aim(f0, 0., 0., kRefNm, vc) ) return 1e30;
    Ray chief = makeRay(f0, vc[0], vc[1]);
    if ( !trace(chief, kRefNm, false, n - 1) ) return 1e30;
    const double nImg = index(n - 1, kRefNm);
    const double tE = (_xpZ - chief.p.z) / chief.d.z;
    const Vec3 E = add(chief.p, chief.d, tE);
    const double R = std::fabs(Q.z - E.z);
    const double ref = chief.opl + nImg * tE;
    const int rings = 24;
    double s = 0., s2 = 0.;
    int cnt = 0;
    double v[2] = { vc[0], vc[1] };
    for (int i = 0; i < rings; ++i) {
        const double rho = std::sqrt( (i + 0.5) / rings );
        if ( !aim(f0, 0., rho * _stopR, kRefNm, v, i > 0) ) {
            if ( !aim(f0, 0., rho * _stopR, kRefNm, v) ) continue;
        }
        Ray r = makeRay(f0, v[0], v[1]);
        if ( !trace(r, kRefNm, false, n - 1) ) continue;
        const Vec3 oc = sub(r.p, Q);
        const double b = dot(r.d, oc), c = dot(oc, oc) - R * R, disc = b * b - c;
        if (disc < 0.) continue;
        const double opd = ref - (r.opl + nImg * (-b - std::sqrt(disc)));
        s += opd;
        s2 += opd * opd;
        ++cnt;
    }
    if (cnt < rings / 2) return 1e30;
    const double m = s / cnt;
    return std::sqrt( std::max(0., s2 / cnt - m * m) );
}

bool RealLens::setup(const LensPrescription& lens, const RealLensSettings& set, std::string* error)
{
    _lens = lens;
    _set = set;
    const int n = (int)_lens.surfaces.size();
    auto fail = [&](const char* m) {
        if (error) *error = m;
        return false;
    };
    if ( (n < 2) || (_lens.stopIndex < 0) ) return fail("the lens needs surfaces and a stop");

    auto layout = [&]() {
        _z.assign(n, 0.);
        _c.assign(n, 0.);
        for (int i = 0; i < n; ++i) {
            _z[i] = i ? _z[i - 1] + _lens.surfaces[i - 1].thickness : 0.;
            _c[i] = _lens.surfaces[i].radius != 0. ? 1. / _lens.surfaces[i].radius : 0.;
        }
    };
    layout();
    std::vector<double> y, u;
    paraxial(kRefNm, 1., 0., 0., y, u);
    if (std::fabs(u[n - 1]) < 1e-300) return fail("afocal lens");
    double efl = -1. / u[n - 1];
    // scale to the focal length asked
    if (set.focalMm > 0.) {
        const double s = set.focalMm / std::fabs(efl);
        for (LensSurface& sf : _lens.surfaces) {
            sf.radius *= s;
            sf.thickness *= s;
            sf.semiDiameter *= s;
            for (size_t k = 0; k < sf.asph.size(); ++k) {
                sf.asph[k] *= std::pow(s, -(3. + 2. * k));   // a r^(4+2k): sag scales with s
            }
        }
        layout();
        paraxial(kRefNm, 1., 0., 0., y, u);
        efl = -1. / u[n - 1];
    }
    _efl = efl;
    _bfl = -y[n - 1] / u[n - 1];
    const int st = _lens.stopIndex;
    const double a = y[st];                 // stop height of a parallel ray of unit height
    std::vector<double> yb, ub;
    paraxial(kRefNm, 0., 1., 0., yb, ub);
    const double b = yb[st];
    if (std::fabs(a) < 1e-300) return fail("the stop is at a focus");
    _stopGain = a;
    _epZ = b / a;
    // aperture: never wider than the design
    _fNumber = std::max( set.fNumber > 0. ? set.fNumber : _lens.designFNumber, _lens.designFNumber );
    const double designStopR = std::fabs(a) * std::fabs(_efl) / (2. * _lens.designFNumber);
    _stopR = std::fabs(a) * std::fabs(_efl) / (2. * _fNumber);
    _epR = _stopR / std::fabs(a);
    // exit pupil: the paraxial chief ray crosses the axis in image space
    std::vector<double> yc, uc;
    paraxial(kRefNm, -b / a, 1., 0., yc, uc);
    const double zLast = _z[n - 1];
    _xpZ = std::fabs(uc[n - 1]) > 1e-300 ? zLast - yc[n - 1] / uc[n - 1] : zLast;
    // object and image planes
    _infinite = !(set.focusDistanceMm > 0.);
    const double tLast = _lens.surfaces[n - 1].thickness;
    if (_infinite) {
        _zImage = zLast + tLast + set.focusOffsetMm;
        _zObject = -1e30;
    } else {
        _zObject = -set.focusDistanceMm;
        std::vector<double> yo, uo;
        paraxial(kRefNm, 0., 1., _zObject, yo, uo);
        if (std::fabs(uo[n - 1]) < 1e-300) return fail("object at the front focus");
        const double zIm = zLast - yo[n - 1] / uo[n - 1];
        _zImage = zIm + (tLast - _bfl) + set.focusOffsetMm;   // keeps the designer's focus offset
    }
    if (_zImage <= zLast - 1e-9) return fail("the image plane is inside the lens");
    // ideal image height per unit field, and working f-number (paraxial)
    {
        std::vector<double> yk, uk, ym, um;
        if (_infinite) {
            paraxial(kRefNm, 0., 1., _epZ, yk, uk);         // chief ray, tan = 1
            paraxial(kRefNm, _epR, 0., 0., ym, um);         // marginal ray
        } else {
            paraxial(kRefNm, 1., -1. / (_epZ - _zObject), _zObject, yk, uk);
            paraxial(kRefNm, 0., _epR / (_epZ - _zObject), _zObject, ym, um);
        }
        _k = yk[n - 1] + uk[n - 1] * (_zImage - zLast);
        _workingN = 1. / (2. * std::fabs(um[n - 1]));
        if (std::fabs(_k) < 1e-300) return fail("degenerate field");
    }
    _zStart = std::min(0., _epZ) - (std::fabs(_efl) + 1.);
    _sd.assign(n, 0.);   // no barrels yet (the traces below are not clipped)
    if (set.bestFocus) {
        // golden section search of the sensor position with the smallest RMS wavefront on axis,
        // between the paraxial and the marginal focus (spherical aberration moves best focus)
        const double zDesign = _zImage - set.focusOffsetMm;
        double zMarg = zDesign;
        {
            const Field f0 = fieldAt(0., 0.);
            double v[2];
            if ( aim(f0, 0., _stopR, kRefNm, v) ) {
                Ray r = makeRay(f0, v[0], v[1]);
                if ( trace(r, kRefNm, false, n - 1) && (std::fabs(r.d.y) > 1e-300) ) {
                    zMarg = r.p.z - r.p.y * r.d.z / r.d.y;
                }
            }
        }
        double zPar = zDesign;
        {
            std::vector<double> yo, uo;
            if (_infinite) paraxial(kRefNm, 1., 0., 0., yo, uo);
            else paraxial(kRefNm, 0., 1., _zObject, yo, uo);
            zPar = zLast - yo[n - 1] / uo[n - 1];
        }
        const double span = std::max( std::fabs(zPar - zMarg), 1e-4 * std::fabs(_efl) );
        double lo = std::min( std::min(zPar, zMarg), zDesign ) - 0.5 * span;
        double hi = std::max( std::max(zPar, zMarg), zDesign ) + 0.5 * span;
        lo = std::max(lo, zLast + 1e-6 * std::fabs(_efl));
        const double gr = 0.5 * (std::sqrt(5.) - 1.);
        double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
        double f1 = onAxisRmsOpd(x1), f2 = onAxisRmsOpd(x2);
        for (int it = 0; it < 60; ++it) {
            if (f1 < f2) {
                hi = x2; x2 = x1; f2 = f1; x1 = hi - gr * (hi - lo); f1 = onAxisRmsOpd(x1);
            } else {
                lo = x1; x1 = x2; f1 = f2; x2 = lo + gr * (hi - lo); f2 = onAxisRmsOpd(x2);
            }
        }
        _zImage = 0.5 * (lo + hi) + set.focusOffsetMm;
        // the ideal image scale follows the sensor
        std::vector<double> yk, uk;
        if (_infinite) paraxial(kRefNm, 0., 1., _epZ, yk, uk);
        else paraxial(kRefNm, 1., -1. / (_epZ - _zObject), _zObject, yk, uk);
        _k = yk[n - 1] + uk[n - 1] * (_zImage - zLast);
    }

    // automatic semi-diameters: rays of the design aperture over part of the design field
    _sd.assign(n, 0.);
    for (int i = 0; i < n; ++i) {
        _sd[i] = _lens.surfaces[i].semiDiameter;
    }
    bool needAuto = false;
    for (int i = 0; i < n; ++i) {
        needAuto |= (i != st) && (_sd[i] <= 0.);
    }
    if (needAuto) {
        const double userStop = _stopR;
        _stopR = designStopR;
        std::vector<double> mx(n, 0.);
        const double fmax = _lens.designHalfFieldDeg * kPi / 180.;
        const double f0 = std::max(0., std::min(1., set.unvignettedField));
        for (int fi = 0; fi <= 10; ++fi) {
            const double fr = fi / 10.;
            const double ang = fmax * fr;
            // open pupil radius: full up to f0, then shrinking to cornerPupil at the full field
            const double rho = fr <= f0 ? 1. : 1. + (std::max(0.05, set.cornerPupil) - 1.) * (fr - f0) / std::max(1e-6, 1. - f0);
            const double Y = (_infinite ? _k * std::tan(ang) : std::fabs(_efl) * std::tan(ang));
            const Field f = fieldAt(0., Y);
            for (int k = -1; k < 24; ++k) {
                const double t = 2. * kPi * k / 24.;
                const double sx = k < 0 ? 0. : rho * _stopR * std::cos(t), sy = k < 0 ? 0. : rho * _stopR * std::sin(t);
                double v[2];
                if ( !aim(f, sx, sy, kRefNm, v) ) continue;
                Ray r = makeRay(f, v[0], v[1]);
                for (int i = 0; i < n; ++i) {
                    if ( !intersect(i, r) ) break;
                    mx[i] = std::max( mx[i], std::hypot(r.p.x, r.p.y) );
                    // refract to continue
                    const double n1 = index(i - 1, kRefNm), n2 = index(i, kRefNm);
                    if (n1 != n2) {
                        Vec3 N;
                        normalAt(i, r.p.x, r.p.y, N);
                        double cosi = -dot(N, r.d);
                        if (cosi < 0.) { N = Vec3(-N.x, -N.y, -N.z); cosi = -cosi; }
                        const double mu = n1 / n2, kk = 1. - mu * mu * (1. - cosi * cosi);
                        if (kk < 0.) break;
                        r.d = normalize( Vec3(mu * r.d.x + (mu * cosi - std::sqrt(kk)) * N.x,
                                              mu * r.d.y + (mu * cosi - std::sqrt(kk)) * N.y,
                                              mu * r.d.z + (mu * cosi - std::sqrt(kk)) * N.z) );
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if ( (i != st) && (_sd[i] <= 0.) ) {
                _sd[i] = mx[i] > 0. ? 1.01 * mx[i] : 0.;
            }
        }
        _stopR = userStop;
    }
    _sd[st] = _stopR;
    return true;
}

// ============================================================================ RealLensModel

static const std::vector<float>& zernikeBasis(int J, int G)
{
    static std::mutex m;
    static std::vector<float> basis;
    static int bj = 0, bg = 0;
    std::lock_guard<std::mutex> lock(m);
    if ( (bj != J) || (bg != G) ) {
        basis.assign( (size_t)(J + 1) * G * G, 0.f );
        for (int j = 1; j <= J; ++j) {
            for (int y = 0; y < G; ++y) {
                for (int x = 0; x < G; ++x) {
                    const double qx = -1. + 2. * x / (G - 1), qy = -1. + 2. * y / (G - 1);
                    basis[( (size_t)j * G + y ) * G + x] = (float)zernikeNoll(j, qx, qy);
                }
            }
        }
        bj = J;
        bg = G;
    }
    return basis;
}

double RealLensModel::imageHalfFieldDeg() const
{
    return std::atan( _rmax / std::fabs(_lens.efl()) ) * 180. / kPi;
}

RealLensModel::FieldWavefront RealLensModel::fieldWavefront(double r, double nm, bool withMask) const
{
    FieldWavefront out;
    const RealLens& L = _lens;
    const int nS = L.surfaceCount();
    const RealLens::Field f = L.fieldAt(0., r);
    const double R0 = L.stopRadius();
    // chief ray and the four marginal rays (unclipped)
    double vc[2], vp[4][2];
    if ( !L.aim(f, 0., 0., nm, vc) ) return out;
    const double mp[4][2] = { { 0.5, 0. }, { -0.5, 0. }, { 0., 0.5 }, { 0., -0.5 } };
    for (int k = 0; k < 4; ++k) {
        if ( !L.aim(f, mp[k][0] * R0, mp[k][1] * R0, nm, vp[k]) ) return out;
    }
    const double J[2][2] = { { (vp[0][0] - vp[1][0]), (vp[2][0] - vp[3][0]) },
                             { (vp[0][1] - vp[1][1]), (vp[2][1] - vp[3][1]) } };
    auto guess = [&](double px, double py, double v[2]) {
        v[0] = vc[0] + J[0][0] * px + J[0][1] * py;
        v[1] = vc[1] + J[1][0] * px + J[1][1] * py;
    };
    RealLens::Ray chief = L.makeRay(f, vc[0], vc[1]);
    if ( !L.trace(chief, nm, false, nS - 1) ) return out;
    const double nImg = L.index(nS - 1, nm);
    const double tQ = (L.imageZ() - chief.p.z) / chief.d.z;
    const Vec3 Q = add(chief.p, chief.d, tQ);
    const double tE = (L.exitPupilZ() - chief.p.z) / chief.d.z;
    const Vec3 E = add(chief.p, chief.d, tE);
    const double Rref = std::sqrt( dot(sub(Q, E), sub(Q, E)) );
    if ( !(Rref > 0.) ) return out;
    const double oplRef = chief.opl + nImg * tE;
    // a ray from the last surface to the reference sphere: OPD (lens units), sphere point, landing
    auto toSphere = [&](const RealLens::Ray& ray, double& opd, Vec3& X, Vec3& land) {
        const Vec3 oc = sub(ray.p, Q);
        const double b = dot(ray.d, oc), c = dot(oc, oc) - Rref * Rref;
        const double disc = b * b - c;
        if (disc < 0.) return false;
        const double s = -b - std::sqrt(disc);
        X = add(ray.p, ray.d, s);
        opd = oplRef - (ray.opl + nImg * s);
        land = add(ray.p, ray.d, (L.imageZ() - ray.p.z) / ray.d.z);
        return true;
    };
    // pupil coordinates: projection of the sphere points on the image plane axes
    Vec3 Xm[4];
    for (int k = 0; k < 4; ++k) {
        RealLens::Ray ray = L.makeRay(f, vp[k][0], vp[k][1]);
        double opd;
        Vec3 land;
        if ( !L.trace(ray, nm, false, nS - 1) || !toSphere(ray, opd, Xm[k], land) ) return out;
    }
    const double A[4] = { (Xm[0].x - Xm[1].x), (Xm[2].x - Xm[3].x),
                          (Xm[0].y - Xm[1].y), (Xm[2].y - Xm[3].y) };
    const double T = A[0] * A[0] + A[1] * A[1] + A[2] * A[2] + A[3] * A[3];
    const double D = A[0] * A[3] - A[1] * A[2];
    double rho0 = std::sqrt( 0.5 * (T + std::sqrt( std::max(0., T * T - 4. * D * D) )) );   // largest singular value
    if ( !(rho0 > 0.) || (std::fabs(D) < 1e-300) ) return out;
    // the exit pupil must fit in the unit circle of q: also measure the traced rim
    for (int k = 0; k < 16; ++k) {
        const double t = 2. * kPi * k / 16.;
        double v[2];
        guess(std::cos(t), std::sin(t), v);
        if ( !L.aim(f, std::cos(t) * R0, std::sin(t) * R0, nm, v, true) ) continue;
        RealLens::Ray ray = L.makeRay(f, v[0], v[1]);
        double opd;
        Vec3 X, land;
        if ( !L.trace(ray, nm, false, nS - 1) || !toSphere(ray, opd, X, land) ) continue;
        rho0 = std::max( rho0, 1.0001 * std::hypot(X.x - E.x, X.y - E.y) );
    }
    for (int i = 0; i < 4; ++i) {
        out.pupil[i] = A[i] / rho0;
    }
    out.fNumber = Rref / (2. * rho0) / nImg;

    // ---- OPD samples over the stop (unclipped), fit with Zernike polynomials of q
    const int S = 25, Jn = kTerms;
    std::vector<double> qx, qy, w, ex, ey;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            const double px = -1. + 2. * (i + 0.5) / S, py = -1. + 2. * (j + 0.5) / S;
            if (px * px + py * py > 1.) continue;
            double v[2];
            guess(px, py, v);
            if ( !L.aim(f, px * R0, py * R0, nm, v, true) ) continue;
            RealLens::Ray ray = L.makeRay(f, v[0], v[1]);
            double opd;
            Vec3 X, land;
            if ( !L.trace(ray, nm, false, nS - 1) || !toSphere(ray, opd, X, land) ) continue;
            qx.push_back( (X.x - E.x) / rho0 );
            qy.push_back( (X.y - E.y) / rho0 );
            w.push_back(opd * 1000.);          // micrometres
            ex.push_back(land.x - Q.x);
            ey.push_back(land.y - Q.y);
        }
    }
    if ( (int)w.size() < 2 * Jn ) return out;
    std::vector<double> M( (size_t)Jn * Jn, 0. ), rhs(Jn, 0.), z(Jn);
    for (size_t s = 0; s < w.size(); ++s) {
        for (int j = 0; j < Jn; ++j) z[j] = zernikeNoll(j + 1, qx[s], qy[s]);
        for (int a = 0; a < Jn; ++a) {
            rhs[a] += z[a] * w[s];
            for (int b = a; b < Jn; ++b) M[(size_t)a * Jn + b] += z[a] * z[b];
        }
    }
    for (int a = 0; a < Jn; ++a) for (int b = 0; b < a; ++b) M[(size_t)a * Jn + b] = M[(size_t)b * Jn + a];
    if ( !solve(M, rhs, Jn) ) return out;
    out.coef.assign(Jn + 1, 0.);
    for (int j = 0; j < Jn; ++j) out.coef[j + 1] = rhs[j];
    // residual, RMS without piston and tilt, gradient consistency
    double res = 0., rms = 0., mean = 0., sxy = 0., sgg = 0.;
    auto fit = [&](double x, double y, int from) {
        double v = 0.;
        for (int j = from; j <= Jn; ++j) v += out.coef[j] * zernikeNoll(j, x, y);
        return v;
    };
    std::vector<double> wt(w.size());
    for (size_t s = 0; s < w.size(); ++s) {
        const double full = fit(qx[s], qy[s], 1);
        res += (full - w[s]) * (full - w[s]);
        wt[s] = w[s] - (full - fit(qx[s], qy[s], 4));   // remove the fitted piston and tilt
        mean += wt[s];
        const double e = 1e-4;
        const double gx = (fit(qx[s] + e, qy[s], 2) - fit(qx[s] - e, qy[s], 2)) / (2. * e) * 1e-3;   // lens units
        const double gy = (fit(qx[s], qy[s] + e, 2) - fit(qx[s], qy[s] - e, 2)) / (2. * e) * 1e-3;
        const double px = -2. * out.fNumber * gx, py = -2. * out.fNumber * gy;
        sxy += px * ex[s] + py * ey[s];
        sgg += px * px + py * py;
    }
    mean /= w.size();
    for (size_t s = 0; s < w.size(); ++s) rms += (wt[s] - mean) * (wt[s] - mean);
    out.rmsUm = std::sqrt(rms / w.size());
    out.fitResidualUm = std::sqrt(res / w.size());
    out.gradientCheck = sgg > 0. ? sxy / sgg : 1.;

    // ---- vignetting mask over q (rays clipped by the barrels; the stop is tested separately)
    if (withMask) {
        const int G = kMask;
        out.mask.assign( (size_t)G * G, 0.f );
        const double inv = 1. / (out.pupil[0] * out.pupil[3] - out.pupil[1] * out.pupil[2]);
        const double P[4] = { out.pupil[3] * inv, -out.pupil[1] * inv, -out.pupil[2] * inv, out.pupil[0] * inv };   // q -> p
        int inside = 0, open = 0;
        for (int j = 0; j < G; ++j) {
            for (int i = 0; i < G; ++i) {
                const double x = -1. + 2. * i / (G - 1), y = -1. + 2. * j / (G - 1);
                const double px = P[0] * x + P[1] * y, py = P[2] * x + P[3] * y;
                const double pr = std::hypot(px, py);
                if (pr > 1.05) continue;
                double v[2];
                guess(px, py, v);
                if ( !L.aim(f, px * R0, py * R0, nm, v, true) ) continue;
                RealLens::Ray ray = L.makeRay(f, v[0], v[1]);
                int blocked = -1;
                if ( !L.trace(ray, nm, false, -1, &blocked) ) continue;
                if (blocked < 0) {
                    out.mask[(size_t)j * G + i] = 1.f;
                }
                if (pr <= 1.) {
                    ++inside;
                    open += blocked < 0;
                }
            }
        }
        const double Tm = inside ? (double)open / inside : 0.;
        const double cosT = chief.d.z;
        out.illum = Tm * std::fabs(D) * cosT * cosT / (Rref * Rref);
    }
    out.ok = true;
    return out;
}

bool RealLensModel::prepare(const ImageGeometry& g, const std::vector<double>& nm,
                            std::function<void(int, int)> progress, std::string* error)
{
    _nm = nm;
    const int nl = (int)nm.size();
    _refL = 0;
    for (int l = 0; l < nl; ++l) {
        if (std::fabs(nm[l] - kRefNm) < std::fabs(nm[_refL] - kRefNm)) _refL = l;
    }
    // sensor radius of the image corner (desqueezed), in lens units (mm)
    const double cx = g.centerX * g.width, cy = g.centerY * g.height, ax = g.pixelAspect;
    double rpx = 0.;
    const double corners[4][2] = { { 0., 0. }, { (double)g.width, 0. }, { 0., (double)g.height }, { (double)g.width, (double)g.height } };
    for (int i = 0; i < 4; ++i) rpx = std::max( rpx, std::hypot( (corners[i][0] - cx) * ax, corners[i][1] - cy ) );
    _pitchUm = g.pixelPitchUm();
    _rmax = rpx * _pitchUm * 1e-3;

    const int nPsf = 21, nDist = 97;
    _rPsf.resize(nPsf);
    for (int k = 0; k < nPsf; ++k) _rPsf[k] = 1.02 * _rmax * k / (nPsf - 1);
    _rDist.resize(nDist);
    for (int k = 0; k < nDist; ++k) _rDist[k] = 1.04 * _rmax * k / (nDist - 1);
    // Wavelengths actually traced: all of them up to 9 (by default), else 9 spread over the range; the
    // others are interpolated (cubic in the wavelength: the dispersion of glasses is smooth).
    std::vector<double> nmT;
    const int maxT = std::max(4, _maxTraced);
    if (nl <= maxT) {
        nmT = nm;
    } else {
        // spread evenly in 1 / lambda^2, the variable in which the index of glasses is smoothest (Cauchy)
        const double u0 = 1. / (nm.front() * nm.front()), u1 = 1. / (nm.back() * nm.back());
        for (int i = 0; i < maxT; ++i) nmT.push_back( 1. / std::sqrt( u0 + (u1 - u0) * i / (maxT - 1.) ) );
        nmT.front() = nm.front();
        nmT.back() = nm.back();
        std::sort(nmT.begin(), nmT.end());
    }
    const int nT = (int)nmT.size();
    _tracedWavelengths = nT;
    int refT = 0;
    for (int i = 0; i < nT; ++i) {
        if (std::fabs(nmT[i] - kRefNm) < std::fabs(nmT[refT] - kRefNm)) refT = i;
    }
    std::vector<std::vector<FieldWavefront> > wfT( nPsf, std::vector<FieldWavefront>(nT) );
    std::vector<std::vector<double> > distT( nT, std::vector<double>(nDist, std::nan("")) );

    // tasks: wavefronts (radius, wavelength) and chief ray tables (wavelength)
    const int nTasks = nPsf * nT + nT;
    std::atomic<int> next(0), done(0);
    std::mutex pm;
    auto worker = [&]() {
        for (;;) {
            const int t = next++;
            if (t >= nTasks) return;
            if (t < nPsf * nT) {
                const int k = t / nT, l = t % nT;
                wfT[k][l] = fieldWavefront(_rPsf[k], nmT[l], l == refT);
            } else {
                const int l = t - nPsf * nT;
                for (int k = 0; k < nDist; ++k) {
                    double x, y;
                    if ( _lens.chiefLanding(0., _rDist[k], nmT[l], x, y) ) distT[l][k] = y - _rDist[k];
                }
            }
            const int d = ++done;
            if (progress) {
                std::lock_guard<std::mutex> lock(pm);
                progress(d, nTasks);
            }
        }
    };
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned i = 1; i < nt; ++i) th.emplace_back(worker);
    worker();
    for (std::thread& t : th) t.join();

    if ( !wfT[0][refT].ok ) {
        if (error) *error = "the lens cannot be traced on axis";
        return false;
    }
    // missing samples (field beyond what the lens can image): nearest valid one, no light
    _illum.assign(nPsf, 0.);
    const double i0 = wfT[0][refT].illum;
    for (int k = 0; k < nPsf; ++k) {
        _illum[k] = (wfT[k][refT].ok && i0 > 0.) ? wfT[k][refT].illum / i0 : 0.;
    }
    for (int l = 0; l < nT; ++l) {
        for (int k = 0; k < nPsf; ++k) {
            FieldWavefront& f = wfT[k][l];
            if (!f.ok) {
                int kk = k;
                while ( (kk > 0) && !wfT[kk][l].ok ) --kk;
                f = wfT[kk][l];
                if (!f.ok) f = wfT[k][refT];
            }
        }
        // chief ray table: extrapolate linearly past the last valid sample
        std::vector<double>& d = distT[l];
        int lastOk = -1;
        for (int k = 0; k < nDist; ++k) if ( !std::isnan(d[k]) ) lastOk = k;
        if (lastOk < 0) {
            std::fill(d.begin(), d.end(), 0.);
            continue;
        }
        for (int k = 0; k < nDist; ++k) {
            if ( std::isnan(d[k]) ) {
                if (k < lastOk) {
                    d[k] = d[k - 1];
                } else {
                    const double slope = lastOk > 0 ? (d[lastOk] - d[lastOk - 1]) : 0.;
                    d[k] = d[lastOk] + slope * (k - lastOk);
                }
            }
        }
    }
    // render wavelengths: Lagrange interpolation in 1 / lambda^2 over the (up to) 4 nearest traced wavelengths
    _wf.assign( nPsf, std::vector<FieldWavefront>(nl) );
    _dist.assign( nl, std::vector<double>(nDist, 0.) );
    for (int l = 0; l < nl; ++l) {
        std::vector<int> idx;
        std::vector<double> wgt;
        int exact = -1;
        for (int i = 0; i < nT; ++i) if (std::fabs(nmT[i] - nm[l]) < 1e-9) exact = i;
        if (exact >= 0) {
            idx.push_back(exact);
            wgt.push_back(1.);
        } else {
            int hi = 0;
            while ( (hi < nT) && (nmT[hi] < nm[l]) ) ++hi;
            int i0 = std::max(0, std::min(hi - 2, nT - 4));
            const int cnt = std::min(4, nT);
            if (nT < 4) i0 = 0;
            for (int i = i0; i < i0 + cnt; ++i) {
                double w = 1.;
                auto u = [](double x) { return 1. / (x * x); };
                for (int j = i0; j < i0 + cnt; ++j) {
                    if (j != i) w *= (u(nm[l]) - u(nmT[j])) / (u(nmT[i]) - u(nmT[j]));
                }
                idx.push_back(i);
                wgt.push_back(w);
            }
        }
        for (int k = 0; k < nPsf; ++k) {
            FieldWavefront f = wfT[k][idx[0]];
            f.coef.assign(kTerms + 1, 0.);
            f.fNumber = 0.;
            for (int i = 0; i < 4; ++i) f.pupil[i] = 0.;
            for (size_t q = 0; q < idx.size(); ++q) {
                const FieldWavefront& s = wfT[k][idx[q]];
                for (int j = 0; j <= kTerms; ++j) f.coef[j] += wgt[q] * s.coef[j];
                f.fNumber += wgt[q] * s.fNumber;
                for (int i = 0; i < 4; ++i) f.pupil[i] += wgt[q] * s.pupil[i];
            }
            f.mask = wfT[k][refT].mask;   // vignetting hardly depends on the wavelength
            _wf[k][l] = f;
        }
        for (int k = 0; k < nDist; ++k) {
            double v = 0.;
            for (size_t q = 0; q < idx.size(); ++q) v += wgt[q] * distT[idx[q]][k];
            _dist[l][k] = v;
        }
    }
    _basis = std::make_shared<std::vector<float> >( zernikeBasis(kTerms, kGrid) );
    return true;
}

void RealLensModel::psfInputs(double hx, double hy, int l, WavefrontParams& w, ApertureParams& a) const
{
    const RealLensSettings& set = _lens.settings();
    const double r = std::hypot(hx, hy) * _rmax;
    const double alpha = r > 1e-12 ? std::atan2(hy, hx) - 0.5 * kPi : 0.;
    const int nPsf = (int)_rPsf.size();
    double u = r / std::max(1e-300, _rPsf[1]);
    int k = std::min( (int)u, nPsf - 2 );
    double t = std::min(1., u - k);
    const FieldWavefront& f0 = _wf[k][l];
    const FieldWavefront& f1 = _wf[k + 1][l];
    // interpolated wavefront coefficients (micrometres), with the artistic gains
    std::vector<double> c(kTerms + 1, 0.);
    for (int j = 1; j <= kTerms; ++j) {
        c[j] = set.aberrationGain * ( (1. - t) * f0.coef[j] + t * f1.coef[j] );
    }
    c[4] += set.defocusWaves * 0.55 / (2. * std::sqrt(3.));   // W020 rho^2 = W020/2 + W020/(2 sqrt 3) Z4
    // rotate to the field azimuth: pairs (cos, sin) of the same order and frequency
    for (int j = 2; j <= kTerms; ++j) {
        int n, m;
        zernikeNollNM(j, n, m);
        if (m <= 0) continue;
        // partner with -m
        for (int jj = 2; jj <= kTerms; ++jj) {
            int n2, m2;
            zernikeNollNM(jj, n2, m2);
            if ( (n2 == n) && (m2 == -m) ) {
                const double ca_ = std::cos(m * alpha), sa = std::sin(m * alpha);
                const double cc = c[j], cs = c[jj];
                c[j] = cc * ca_ - cs * sa;
                c[jj] = cc * sa + cs * ca_;
                break;
            }
        }
    }
    const int G = kGrid;
    std::shared_ptr<std::vector<float> > grid = std::make_shared<std::vector<float> >( (size_t)G * G, 0.f );
    const std::vector<float>& B = *_basis;
    for (int j = 2; j <= kTerms; ++j) {
        const float cj = (float)c[j];
        if (cj == 0.f) continue;
        const float* bj = &B[(size_t)j * G * G];
        float* g = grid->data();
        for (size_t p = 0; p < (size_t)G * G; ++p) g[p] += cj * bj[p];
    }
    w = WavefrontParams();
    w.fNumber = (1. - t) * f0.fNumber + t * f1.fNumber;
    w.lambda0Um = 1.;
    w.opdGrid = grid;
    w.opdN = G;

    // pupil: interpolated pupil map and mask, rotated to the field azimuth
    double Aq[4];
    for (int i = 0; i < 4; ++i) Aq[i] = (1. - t) * f0.pupil[i] + t * f1.pupil[i];
    const double det = Aq[0] * Aq[3] - Aq[1] * Aq[2];
    const double P[4] = { Aq[3] / det, -Aq[1] / det, -Aq[2] / det, Aq[0] / det };   // q -> p (unrotated)
    const double ca_ = std::cos(alpha), sa = std::sin(alpha);
    // pupilMap = R(alpha) P R(-alpha)
    const double Rm[4] = { ca_, -sa, sa, ca_ }, Rmi[4] = { ca_, sa, -sa, ca_ };
    double T1[4];
    T1[0] = P[0] * Rmi[0] + P[1] * Rmi[2]; T1[1] = P[0] * Rmi[1] + P[1] * Rmi[3];
    T1[2] = P[2] * Rmi[0] + P[3] * Rmi[2]; T1[3] = P[2] * Rmi[1] + P[3] * Rmi[3];
    a = ApertureParams();
    a.blades = set.blades;
    a.roundness = set.roundness;
    a.rotationDeg = set.rotationDeg;
    a.hasPupilMap = true;
    a.pupilMap[0] = Rm[0] * T1[0] + Rm[1] * T1[2]; a.pupilMap[1] = Rm[0] * T1[1] + Rm[1] * T1[3];
    a.pupilMap[2] = Rm[2] * T1[0] + Rm[3] * T1[2]; a.pupilMap[3] = Rm[2] * T1[1] + Rm[3] * T1[3];
    const int M = kMask;
    if ( !f0.mask.empty() && !f1.mask.empty() ) {
        std::shared_ptr<std::vector<float> > mask = std::make_shared<std::vector<float> >( (size_t)M * M, 0.f );
        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < M; ++i) {
                const double x = -1. + 2. * i / (M - 1), y = -1. + 2. * j / (M - 1);
                // q in the unrotated pupil: R(-alpha) q
                const double ux = ca_ * x + sa * y, uy = -sa * x + ca_ * y;
                const double fu = (ux + 1.) * 0.5 * (M - 1), fv = (uy + 1.) * 0.5 * (M - 1);
                if ( (fu < 0.) || (fv < 0.) || (fu > M - 1) || (fv > M - 1) ) continue;
                const int i0 = std::min( (int)fu, M - 2 ), j0 = std::min( (int)fv, M - 2 );
                const double du = fu - i0, dv = fv - j0;
                auto at = [&](const std::vector<float>& m) {
                    return (1. - dv) * ( (1. - du) * m[(size_t)j0 * M + i0] + du * m[(size_t)j0 * M + i0 + 1] ) +
                           dv * ( (1. - du) * m[(size_t)(j0 + 1) * M + i0] + du * m[(size_t)(j0 + 1) * M + i0 + 1] );
                };
                (*mask)[(size_t)j * M + i] = (float)( (1. - t) * at(f0.mask) + t * at(f1.mask) );
            }
        }
        a.mask = mask;
        a.maskN = M;
    }
}

void RealLensModel::displacementUm(double hx, double hy, int l, double& dx, double& dy) const
{
    const RealLensSettings& set = _lens.settings();
    const double h = std::hypot(hx, hy);
    const double r = h * _rmax;
    const int n = (int)_rDist.size();
    const double u = r / std::max(1e-300, _rDist[1]);
    const int k = std::min( (int)u, n - 2 );
    const double t = u - k;
    auto at = [&](int ll) { return (1. - t) * _dist[ll][k] + t * _dist[ll][k + 1]; };
    const double dref = at(_refL);
    const double d = set.distortionGain * dref + set.lateralColourGain * (at(l) - dref);   // lens units, radial
    if (h < 1e-12) {
        dx = dy = 0.;
        return;
    }
    dx = d * 1000. * hx / h;
    dy = d * 1000. * hy / h;
}

double RealLensModel::illumination(double hx, double hy) const
{
    const double r = std::hypot(hx, hy) * _rmax;
    const int n = (int)_rPsf.size();
    const double u = r / std::max(1e-300, _rPsf[1]);
    const int k = std::min( (int)u, n - 2 );
    const double t = std::min(1., u - k);
    return std::max(0., (1. - t) * _illum[k] + t * _illum[k + 1]);
}

} // namespace ca
