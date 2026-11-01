// ChromaAberrations_OFX - RGB <-> spectrum conversion for the renderer
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "spectralbasis.h"

#include <algorithm>
#include <cmath>

namespace ca {

const char* colorSpaceName(ColorSpace cs)
{
    switch (cs) {
    case eColorSpaceRec709: return "rec709";
    case eColorSpaceRec2020: return "rec2020";
    default: return "acescg";
    }
}

bool colorSpaceFromName(const std::string& name, ColorSpace& cs)
{
    if (name == "acescg" || name == "aces" || name == "ap1") { cs = eColorSpaceACEScg; return true; }
    if (name == "rec709" || name == "srgb" || name == "linear") { cs = eColorSpaceRec709; return true; }
    if (name == "rec2020") { cs = eColorSpaceRec2020; return true; }
    return false;
}

static bool invert3(const double m[9], double r[9])
{
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::fabs(det) < 1e-300) {
        return false;
    }
    const double k = 1. / det;
    r[0] = (e * i - f * h) * k; r[1] = (c * h - b * i) * k; r[2] = (b * f - c * e) * k;
    r[3] = (f * g - d * i) * k; r[4] = (a * i - c * g) * k; r[5] = (c * d - a * f) * k;
    r[6] = (d * h - e * g) * k; r[7] = (b * g - a * h) * k; r[8] = (a * e - b * d) * k;
    return true;
}

void rgbToXyzMatrix(ColorSpace cs, double m[9])
{
    // chromaticities of R, G, B and white
    double p[8];
    switch (cs) {
    case eColorSpaceRec709: {
        const double q[8] = { 0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290 };
        std::copy(q, q + 8, p);
        break;
    }
    case eColorSpaceRec2020: {
        const double q[8] = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290 };
        std::copy(q, q + 8, p);
        break;
    }
    default: {
        const double q[8] = { 0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767 };
        std::copy(q, q + 8, p);
        break;
    }
    }
    double prim[9];   // columns: XYZ of each primary with Y = 1
    for (int c = 0; c < 3; ++c) {
        const double x = p[2 * c], y = p[2 * c + 1];
        prim[0 * 3 + c] = x / y;
        prim[1 * 3 + c] = 1.;
        prim[2 * 3 + c] = (1. - x - y) / y;
    }
    const double wx = p[6] / p[7], wy = 1., wz = (1. - p[6] - p[7]) / p[7];
    double inv[9];
    invert3(prim, inv);
    const double s[3] = { inv[0] * wx + inv[1] * wy + inv[2] * wz,
                          inv[3] * wx + inv[4] * wy + inv[5] * wz,
                          inv[6] * wx + inv[7] * wy + inv[8] * wz };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m[r * 3 + c] = prim[r * 3 + c] * s[c];
        }
    }
}

// Solves A x = b (n x n, row-major) by Gaussian elimination with partial pivoting.
static bool solveDense(std::vector<double>& A, std::vector<double>& b, int n)
{
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(A[(size_t)i * n + k]) > std::fabs(A[(size_t)piv * n + k])) {
                piv = i;
            }
        }
        if (std::fabs(A[(size_t)piv * n + k]) < 1e-14) {
            return false;
        }
        if (piv != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[(size_t)k * n + j], A[(size_t)piv * n + j]);
            }
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = A[(size_t)i * n + k] / A[(size_t)k * n + k];
            if (f == 0.) {
                continue;
            }
            for (int j = k; j < n; ++j) {
                A[(size_t)i * n + j] -= f * A[(size_t)k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) {
            s -= A[(size_t)i * n + j] * b[j];
        }
        b[i] = s / A[(size_t)i * n + i];
    }
    return true;
}

// Smoothest basis under the equality constraints; the entries flagged in 'zero' are pulled
// towards 0 by a strong penalty (a hard constraint is not always feasible: the green primary
// of ACEScg lies outside the spectral locus, so it needs some negative light).
static bool solveBasis(int n, const std::vector<double> w[3], const std::vector<char>& zero, std::vector<double>& x)
{
    const int nu = 3 * n;
    // constraints: response_c . B_d = delta_cd for d = 0, 1 (d = 2 follows from the sums),
    // sum_d B_d(i) = 1, and the forced zeros
    std::vector<std::vector<double> > rows;
    std::vector<double> rhs;
    for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 2; ++d) {
            std::vector<double> r(nu, 0.);
            for (int i = 0; i < n; ++i) {
                r[d * n + i] = w[c][i];
            }
            rows.push_back(r);
            rhs.push_back(c == d ? 1. : 0.);
        }
    }
    for (int i = 0; i < n; ++i) {
        std::vector<double> r(nu, 0.);
        r[i] = r[n + i] = r[2 * n + i] = 1.;
        rows.push_back(r);
        rhs.push_back(1.);
    }
    const int m = (int)rows.size();
    const int t = nu + m;
    std::vector<double> K( (size_t)t * t, 0. ), b(t, 0.);
    // objective: second differences (smoothness), plus a tiny ridge for uniqueness
    for (int d = 0; d < 3; ++d) {
        for (int i = 1; i + 1 < n; ++i) {
            const int idx[3] = { d * n + i - 1, d * n + i, d * n + i + 1 };
            const double c[3] = { 1., -2., 1. };
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    K[(size_t)idx[p] * t + idx[q]] += c[p] * c[q];
                }
            }
        }
    }
    for (int u = 0; u < nu; ++u) {
        K[(size_t)u * t + u] += zero[u] ? 1e3 : 1e-6;
    }
    for (int r = 0; r < m; ++r) {
        for (int u = 0; u < nu; ++u) {
            K[(size_t)(nu + r) * t + u] = rows[r][u];
            K[(size_t)u * t + nu + r] = rows[r][u];
        }
        b[nu + r] = rhs[r];
    }
    if ( !solveDense(K, b, t) ) {
        return false;
    }
    x.assign(b.begin(), b.begin() + nu);
    return true;
}

// Smoothest spectrum s with response . s = target exactly; entries that come out negative are
// pulled towards 0 by a strong penalty, repeatedly (they stay negative only when the colour lies
// outside the spectral locus, like the green primary of ACEScg).
static void smoothestSpectrum(int n, const std::vector<double> resp[3], const double target[3], std::vector<double>& s)
{
    std::vector<char> pen(n, 0);
    s.assign(n, 0.);
    for (int iter = 0; iter <= n; ++iter) {
        const int t = n + 3;
        std::vector<double> K( (size_t)t * t, 0. ), b(t, 0.);
        for (int i = 1; i + 1 < n; ++i) {
            const int idx[3] = { i - 1, i, i + 1 };
            const double c[3] = { 1., -2., 1. };
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    K[(size_t)idx[p] * t + idx[q]] += c[p] * c[q];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            K[(size_t)i * t + i] += pen[i] ? 1e3 : 1e-6;
        }
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < n; ++i) {
                K[(size_t)(n + c) * t + i] = resp[c][i];
                K[(size_t)i * t + n + c] = resp[c][i];
            }
            b[n + c] = target[c];
        }
        if ( !solveDense(K, b, t) ) {
            break;
        }
        s.assign(b.begin(), b.begin() + n);
        bool added = false;
        for (int i = 0; i < n; ++i) {
            if (!pen[i] && s[i] < -1e-9) {
                pen[i] = 1;
                added = true;
            }
        }
        if (!added) {
            break;
        }
    }
}

// CIE observer in the working space at one wavelength (not normalised)
static void cieWorking(const double minv[9], double nm, double out[3])
{
    double X, Y, Z;
    cieXYZ(nm, X, Y, Z);
    for (int c = 0; c < 3; ++c) {
        out[c] = minv[c * 3 + 0] * X + minv[c * 3 + 1] * Y + minv[c * 3 + 2] * Z;
    }
}

double fitCameraMatrix(const CameraSensitivity& cam, ColorSpace cs, double loNm, double hiNm, double m[9])
{
    double x2r[9], minv[9];
    rgbToXyzMatrix(cs, x2r);
    invert3(x2r, minv);
    // 5 nm grid; camera channels and CIE working channels both normalised to 1 for a flat spectrum
    std::vector<double> nm;
    for (double l = std::ceil(loNm / 5.) * 5.; l <= hiNm + 1e-9; l += 5.) nm.push_back(l);
    const int n = (int)nm.size();
    std::vector<double> camS[3], cie[3];
    for (int c = 0; c < 3; ++c) {
        camS[c].resize(n);
        cie[c].resize(n);
    }
    double sc[3] = { 0., 0., 0. }, st[3] = { 0., 0., 0. };
    for (int i = 0; i < n; ++i) {
        double w[3];
        cieWorking(minv, nm[i], w);
        for (int c = 0; c < 3; ++c) {
            camS[c][i] = cam.at(c, nm[i]);
            cie[c][i] = w[c];
            sc[c] += camS[c][i];
            st[c] += w[c];
        }
    }
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n; ++i) {
            camS[c][i] /= std::max(sc[c], 1e-30);
            cie[c][i] /= st[c];
        }
    }
    // training colours: 190 reflectances under a flat light, plus white (heavily weighted)
    std::vector<std::vector<double> > A, Y;   // [sample][3]
    std::vector<double> W;
    const int np = trainingPatchCount();
    for (int p = 0; p <= np; ++p) {
        std::vector<double> a(3, 0.), y(3, 0.);
        for (int i = 0; i < n; ++i) {
            double rho = 1.;
            if (p < np) {
                const float* r = trainingPatch(p);
                const double u = (nm[i] - 380.) / 5.;
                const int k = std::max(0, std::min(79, (int)u));
                rho = r[k] + (u - k) * (r[k + 1] - r[k]);
            }
            for (int c = 0; c < 3; ++c) {
                a[c] += camS[c][i] * rho;
                y[c] += cie[c][i] * rho;
            }
        }
        A.push_back(a);
        Y.push_back(y);
        W.push_back(p < np ? 1. : 1e4);
    }
    // weighted least squares, one row of the matrix per output channel
    double N[9] = { 0. };
    for (size_t k = 0; k < A.size(); ++k)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) N[i * 3 + j] += W[k] * A[k][i] * A[k][j];
    double Ni[9];
    if (!invert3(N, Ni)) {
        for (int i = 0; i < 9; ++i) m[i] = (i % 4 == 0) ? 1. : 0.;
        return 1.;
    }
    for (int c = 0; c < 3; ++c) {
        double b[3] = { 0., 0., 0. };
        for (size_t k = 0; k < A.size(); ++k)
            for (int i = 0; i < 3; ++i) b[i] += W[k] * A[k][i] * Y[k][c];
        for (int j = 0; j < 3; ++j) m[c * 3 + j] = Ni[j * 3 + 0] * b[0] + Ni[j * 3 + 1] * b[1] + Ni[j * 3 + 2] * b[2];
    }
    double e2 = 0.;
    for (int k = 0; k < np; ++k) {
        for (int c = 0; c < 3; ++c) {
            const double v = m[c * 3] * A[k][0] + m[c * 3 + 1] * A[k][1] + m[c * 3 + 2] * A[k][2] - Y[k][c];
            e2 += v * v;
        }
    }
    return std::sqrt(e2 / (3. * np));
}

// Response of each output channel to each render wavelength, normalised so that a flat
// spectrum gives (1, 1, 1): CIE observer, or camera sensitivities through the camera matrix
static void channelResponses(const std::vector<WavelengthSample>& wl, ColorSpace cs, std::vector<double> r[3],
                             const CameraSensitivity* cam = 0, const double* camM = 0)
{
    const int n = (int)wl.size();
    double m[9], minv[9];
    rgbToXyzMatrix(cs, m);
    invert3(m, minv);
    for (int c = 0; c < 3; ++c) {
        r[c].assign(n, 0.);
    }
    for (int i = 0; i < n; ++i) {
        double w[3];
        if (cam && camM) {
            const double k[3] = { cam->at(0, wl[i].nm), cam->at(1, wl[i].nm), cam->at(2, wl[i].nm) };
            for (int c = 0; c < 3; ++c) w[c] = camM[c * 3] * k[0] + camM[c * 3 + 1] * k[1] + camM[c * 3 + 2] * k[2];
        } else {
            cieWorking(minv, wl[i].nm, w);
        }
        for (int c = 0; c < 3; ++c) {
            r[c][i] = w[c] * wl[i].weight;
        }
    }
    for (int c = 0; c < 3; ++c) {
        double s = 0.;
        for (int i = 0; i < n; ++i) {
            s += r[c][i];
        }
        for (int i = 0; i < n; ++i) {
            r[c][i] /= s;
        }
    }
}

SpectralBasis buildSpectralBasis(int nWavelengths, ColorSpace cs, double loNm, double hiNm, const CameraSensitivity* camera)
{
    SpectralBasis sb;
    sb.space = cs;
    sb.wavelengths = sampleWavelengths(std::max(3, nWavelengths), loNm, hiNm);
    const int n = sb.count();
    if ( camera && camera->valid() ) {
        // camera matrix: camera channels are scaled to 1 for a flat spectrum over the fit range
        double m[9];
        sb.cameraFitError = fitCameraMatrix(*camera, cs, loNm, hiNm, m);
        double sc[3] = { 0., 0., 0. };
        int k = 0;
        for (double l = std::ceil(loNm / 5.) * 5.; l <= hiNm + 1e-9; l += 5., ++k)
            for (int c = 0; c < 3; ++c) sc[c] += camera->at(c, l);
        for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d) sb.cameraMatrix[c * 3 + d] = m[c * 3 + d] / std::max(sc[d], 1e-30);
        sb.camera = camera->name;
        channelResponses(sb.wavelengths, cs, sb.response, camera, sb.cameraMatrix);
        for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d) sb.cameraMatrix[c * 3 + d] = m[c * 3 + d];
    } else {
        channelResponses(sb.wavelengths, cs, sb.response);
    }

    // 1. Basis of the Rec.709 primaries: they lie well inside the spectral locus, so their
    //    smoothest spectra are (almost) non-negative. Smoothest basis first, then the negative
    //    values are pushed towards zero until none is left.
    std::vector<double> r709[3];
    channelResponses(sb.wavelengths, eColorSpaceRec709, r709);
    std::vector<char> zero(3 * n, 0);
    std::vector<double> x;
    solveBasis(n, r709, zero, x);
    for (int iter = 0; iter < 3 * n; ++iter) {
        bool added = false;
        for (int u = 0; u < 3 * n; ++u) {
            if (!zero[u] && x[u] < -1e-6) {
                zero[u] = 1;
                added = true;
            }
        }
        if (!added) {
            break;
        }
        solveBasis(n, r709, zero, x);
    }

    // 2. Expressed in the working space: B = B709 T with T = (R B709)^-1, so that R B = I
    //    exactly. Greys stay flat (R B709 maps (1,1,1) to (1,1,1)). Colours outside Rec.709
    //    (saturated ACEScg greens, say) get some negative light, as they must.
    double rb[9], t[9];
    for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
            double s = 0.;
            for (int i = 0; i < n; ++i) {
                s += sb.response[c][i] * x[d * n + i];
            }
            rb[c * 3 + d] = s;
        }
    }
    invert3(rb, t);
    for (int d = 0; d < 3; ++d) {
        sb.basis[d].assign(n, 0.);
        for (int i = 0; i < n; ++i) {
            for (int e = 0; e < 3; ++e) {
                sb.basis[d][i] += x[e * n + i] * t[e * 3 + d];
            }
        }
    }

    // 3. Chromaticity table: smoothest spectrum with the exact RGB, negative values pushed to 0
    sb.lutN = 49;
    sb.lutLo = -0.5;
    sb.lutStep = 2. / (sb.lutN - 1);   // r, g in [-0.5, 1.5]
    sb.lut.assign( (size_t)n * sb.lutN * sb.lutN, 0.f );
    for (int j = 0; j < sb.lutN; ++j) {
        for (int i = 0; i < sb.lutN; ++i) {
            const double rgb[3] = { sb.lutLo + i * sb.lutStep, sb.lutLo + j * sb.lutStep, 0. };
            const double target[3] = { rgb[0], rgb[1], 1. - rgb[0] - rgb[1] };
            std::vector<double> spec;
            smoothestSpectrum(n, sb.response, target, spec);
            for (int l = 0; l < n; ++l) {
                sb.lut[( (size_t)l * sb.lutN + j ) * sb.lutN + i] = (float)spec[l];
            }
        }
    }
    return sb;
}

double spectrumOf(const SpectralBasis& sb, int l, double r, double g, double b)
{
    const double sum = r + g + b;
    if ( (sb.lutN > 1) && (sum > 1e-6 * (std::fabs(r) + std::fabs(g) + std::fabs(b))) && (sum > 0.) ) {
        const double u = (r / sum - sb.lutLo) / sb.lutStep, v = (g / sum - sb.lutLo) / sb.lutStep;
        if ( (u >= 0.) && (v >= 0.) && (u <= sb.lutN - 1) && (v <= sb.lutN - 1) ) {
            const int i = std::min( (int)u, sb.lutN - 2 ), j = std::min( (int)v, sb.lutN - 2 );
            const double fu = u - i, fv = v - j;
            const float* t = &sb.lut[(size_t)l * sb.lutN * sb.lutN];
            const double a = t[j * sb.lutN + i], bb = t[j * sb.lutN + i + 1];
            const double c = t[(j + 1) * sb.lutN + i], d = t[(j + 1) * sb.lutN + i + 1];
            return sum * ( (1. - fv) * ( (1. - fu) * a + fu * bb ) + fv * ( (1. - fu) * c + fu * d ) );
        }
    }
    return sb.basis[0][l] * r + sb.basis[1][l] * g + sb.basis[2][l] * b;
}

} // namespace ca
