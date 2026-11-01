// ChromaAberrations_OFX - optical prescriptions (.lens files)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// Format (text, '#' starts a comment):
//   name = Double Gauss f/2
//   source = where the prescription comes from
//   fnumber = 2.0            design f-number (the lens cannot open wider)
//   half_field = 10.8        design half field of view, degrees
//   surfaces
//   <radius> <thickness> <material> [semi-diameter | auto] [stop] [k=<conic>] [a4=.. a6=.. ... a16=..]
//   ...
// One line per surface, from the object side. radius in any unit (inf = flat), thickness to the
// next surface (the last one: to the image plane), material after the surface: air, a catalogue
// glass (N-BK7, SF4@hikari) or nd/vd. The semi-diameter is computed when missing or 'auto'. The
// lens is scaled to the focal length asked, so the unit does not matter.

#pragma once

#include <string>
#include <vector>

#include "glass.h"

namespace ca {

struct LensSurface
{
    double radius = 0.;        // 0 = flat
    double thickness = 0.;
    Glass glass;               // medium after the surface
    double semiDiameter = 0.;  // 0 = automatic
    bool stop = false;
    double conic = 0.;
    std::vector<double> asph;  // a4, a6, a8, ... (sag += a4 r^4 + a6 r^6 + ...)
};

struct LensPrescription
{
    std::string name, source;
    double designFNumber = 0.;
    double designHalfFieldDeg = 0.;
    std::vector<LensSurface> surfaces;
    int stopIndex = -1;
};

bool parseLensText(const std::string& text, LensPrescription& lens, std::string* error = 0);
bool loadLensFile(const std::string& path, LensPrescription& lens, std::string* error = 0);

// Lenses compiled into the plugin (the files of the lenses/ folder)
int builtinLensCount();
const char* builtinLensId(int i);      // file name without extension, e.g. "double_gauss_f2"
const char* builtinLensText(int i);
bool loadBuiltinLens(const std::string& id, LensPrescription& lens, std::string* error = 0);

} // namespace ca
