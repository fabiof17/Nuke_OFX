// ChromaAberrations_OFX - optical glasses: refractive index versus wavelength
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "glass.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ca {

namespace {

struct Entry
{
    const char* vendor;
    const char* name;
    int formula;
    int count;
    double c[16];
    double lo, hi;
};

const Entry kCatalog[] = {
#include "glasscatalog.inc"
};

const char* const kVendorOrder[] = { "schott", "ohara", "hoya", "cdgm", "hikari", "sumita" };

std::string upper(std::string s)
{
    for (char& ch : s) ch = (char)std::toupper( (unsigned char)ch );
    return s;
}

std::string lowerStr(std::string s)
{
    for (char& ch : s) ch = (char)std::tolower( (unsigned char)ch );
    return s;
}

} // namespace

int glassCatalogSize()
{
    return (int)( sizeof(kCatalog) / sizeof(kCatalog[0]) );
}

double Glass::index(double nm) const
{
    const double l = nm * 1e-3, l2 = l * l;
    switch (formula) {
    case 1: {   // n^2 - 1 = C1 + sum Ci l^2 / (l^2 - Cj^2)
        double s = c[0];
        for (int i = 1; i + 1 < count; i += 2) {
            s += c[i] * l2 / (l2 - c[i + 1] * c[i + 1]);
        }
        return std::sqrt(1. + s);
    }
    case 2: {   // n^2 - 1 = C1 + sum Ci l^2 / (l^2 - Cj)
        double s = c[0];
        for (int i = 1; i + 1 < count; i += 2) {
            s += c[i] * l2 / (l2 - c[i + 1]);
        }
        return std::sqrt(1. + s);
    }
    case 3: {   // n^2 = C1 + sum Ci l^Cj
        double s = c[0];
        for (int i = 1; i + 1 < count; i += 2) {
            s += c[i] * std::pow(l, c[i + 1]);
        }
        return std::sqrt(s);
    }
    case 4:     // Cauchy n = A + B / l^2 + C / l^4
        return c[0] + c[1] / l2 + c[2] / (l2 * l2);
    default:
        return 1.;
    }
}

double Glass::vd() const
{
    const double nF = index(486.1327), nC = index(656.2725);
    return (nd() - 1.) / std::max(1e-12, nF - nC);
}

Glass modelGlass(double nd, double vd)
{
    // n(l) = A + B / l^2 + C / l^4 with n(d) = nd, nF - nC = (nd - 1) / vd and the partial
    // dispersion (ng - nF) / (nF - nC) of the normal line
    const double ld = 0.5875618, lF = 0.4861327, lC = 0.6562725, lg = 0.4358343;
    const double dFC = (nd - 1.) / std::max(1e-6, vd);
    const double pgF = 0.6438 - 0.001682 * vd;
    auto a2 = [](double l) { return 1. / (l * l); };
    auto a4 = [](double l) { return 1. / (l * l * l * l); };
    // unknowns B, C from: B (a2F - a2C) + C (a4F - a4C) = dFC ; B (a2g - a2F) + C (a4g - a4F) = pgF dFC
    const double m11 = a2(lF) - a2(lC), m12 = a4(lF) - a4(lC);
    const double m21 = a2(lg) - a2(lF), m22 = a4(lg) - a4(lF);
    const double det = m11 * m22 - m12 * m21;
    const double B = (dFC * m22 - m12 * pgF * dFC) / det;
    const double C = (m11 * pgF * dFC - m21 * dFC) / det;
    Glass g;
    g.vendor = "model";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.5f/%.2f", nd, vd);
    g.name = buf;
    g.formula = 4;
    g.count = 3;
    g.c[0] = nd - B * a2(ld) - C * a4(ld);
    g.c[1] = B;
    g.c[2] = C;
    g.loUm = 0.35;
    g.hiUm = 1.;
    return g;
}

bool findGlass(const std::string& specIn, Glass& g, std::string* error)
{
    std::string spec = specIn;
    spec.erase( std::remove_if(spec.begin(), spec.end(), [](char ch) { return std::isspace( (unsigned char)ch ); }), spec.end() );
    if ( spec.empty() || (lowerStr(spec) == "air") ) {
        g = Glass();
        g.name = "air";
        return true;
    }
    const size_t slash = spec.find('/');
    if (slash != std::string::npos) {
        char* e1 = 0;
        char* e2 = 0;
        const std::string a = spec.substr(0, slash), b = spec.substr(slash + 1);
        const double nd = std::strtod(a.c_str(), &e1), vd = std::strtod(b.c_str(), &e2);
        if ( (*e1 == 0) && (*e2 == 0) && (nd > 1.) && (vd > 1.) ) {
            g = modelGlass(nd, vd);
            return true;
        }
    }
    std::string name = upper(spec), vendor;
    const size_t at = name.find('@');
    if (at != std::string::npos) {
        vendor = lowerStr( name.substr(at + 1) );
        name = name.substr(0, at);
    }
    const Entry* best = 0;
    int bestRank = 1000;
    for (const Entry& e : kCatalog) {
        if (upper(e.name) != name) {
            continue;
        }
        if ( !vendor.empty() && (vendor != e.vendor) ) {
            continue;
        }
        int rank = 100;
        for (int i = 0; i < 6; ++i) {
            if (std::string(kVendorOrder[i]) == e.vendor) {
                rank = i;
            }
        }
        if (rank < bestRank) {
            bestRank = rank;
            best = &e;
        }
    }
    if (!best) {
        if (error) {
            *error = "unknown glass '" + specIn + "' (use a catalogue name, name@maker, or nd/vd)";
        }
        return false;
    }
    g = Glass();
    g.vendor = best->vendor;
    g.name = best->name;
    g.formula = best->formula;
    g.count = best->count;
    for (int i = 0; i < best->count && i < 16; ++i) {
        g.c[i] = best->c[i];
    }
    g.loUm = best->lo;
    g.hiUm = best->hi;
    return true;
}

} // namespace ca
