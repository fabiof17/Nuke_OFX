// ChromaAberrations_OFX - optical glasses: refractive index versus wavelength
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// The catalogue holds the dispersion formulas of the Schott, Ohara, Hoya, CDGM, Hikari and
// Sumita glasses (refractiveindex.info database, public domain). A glass can also be given by
// its index nd and Abbe number vd ("model glass", as in patents): its dispersion then follows
// the normal line of optical glasses.

#pragma once

#include <string>

namespace ca {

struct Glass
{
    std::string vendor, name;   // "air" for air
    int formula = 0;            // 0 air, 1 Sellmeier, 2 Sellmeier-2, 3 polynomial, 4 model glass (Cauchy)
    double c[16] = { 0 };
    int count = 0;
    double loUm = 0., hiUm = 0.;

    double index(double nm) const;     // refractive index at a wavelength in nanometres
    double nd() const { return index(587.5618); }
    double vd() const;                 // Abbe number
    bool isAir() const { return formula == 0; }
};

// "air", a catalogue name ("N-BK7", "SF4@hikari" to choose the maker) or "nd/vd" ("1.67/47.2").
// Returns false (and an error) if the glass is unknown.
bool findGlass(const std::string& spec, Glass& g, std::string* error = 0);

// Model glass from nd and vd: Cauchy formula on the normal line (Schott P_gF = 0.6438 - 0.001682 vd)
Glass modelGlass(double nd, double vd);

int glassCatalogSize();

} // namespace ca
