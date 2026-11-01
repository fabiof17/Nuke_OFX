// ChromaAberrations_OFX - parameter files and key=value arguments of the command-line tools
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "psfbank.h"
#include "lensmodel.h"
#include "reallens.h"
#include <memory>

namespace catool {

typedef std::map<std::string, std::string> Params;

inline std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

inline std::string lower(std::string s)
{
    for (char& c : s) c = (char)std::tolower( (unsigned char)c );
    return s;
}

inline bool parseLine(const std::string& line, Params& p)
{
    std::string l = line.substr( 0, line.find('#') );
    const size_t eq = l.find('=');
    if (eq == std::string::npos) {
        return trim(l).empty();
    }
    p[ lower( trim( l.substr(0, eq) ) ) ] = trim( l.substr(eq + 1) );
    return true;
}

// Reads a parameter file; values already in p (command line) win.
inline bool readParamFile(const char* tool, const std::string& path, Params& p)
{
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "%s: cannot read %s\n", tool, path.c_str());
        return false;
    }
    Params fp;
    std::string line;
    int n = 0;
    while ( std::getline(f, line) ) {
        ++n;
        if ( !parseLine(line, fp) ) {
            std::fprintf(stderr, "%s: %s:%d: expected key = value\n", tool, path.c_str(), n);
            return false;
        }
    }
    for (auto& kv : fp) {
        if ( !p.count(kv.first) ) {
            p[kv.first] = kv.second;
        }
    }
    return true;
}

inline double num(const Params& p, const char* k, double def)
{
    auto it = p.find(k);
    return it == p.end() ? def : std::atof( it->second.c_str() );
}

inline std::string str(const Params& p, const char* k, const char* def)
{
    auto it = p.find(k);
    return it == p.end() ? def : it->second;
}

// keys describing the lens, shared by every tool
inline const std::vector<std::string>& lensKeys()
{
    static const std::vector<std::string> k = { "fnumber", "lambda0", "w040", "w131", "w222", "w220", "w311", "w020",
        "axial_ca", "lateral_ca", "spherochromatism", "zernike", "blades", "roundness", "rotation", "aspect",
        "vignetting", "vignetting_start", "pixel_aspect", "filmback", "center_x", "center_y", "method", "max_fft",
        // real lenses
        "lens", "focal", "focus_distance", "focus_offset", "design_focus", "aberration_gain", "distortion_gain",
        "lateral_gain", "unvignetted_field", "corner_pupil",
        // preset files of the plugin
        "name", "hint" };
    return k;
}

inline bool checkKeys(const char* tool, const Params& p, const std::vector<std::string>& extra)
{
    for (auto& kv : p) {
        const std::vector<std::string>& lk = lensKeys();
        if ( (std::find(lk.begin(), lk.end(), kv.first) == lk.end()) &&
             (std::find(extra.begin(), extra.end(), kv.first) == extra.end()) ) {
            std::fprintf(stderr, "%s: unknown key '%s' (see --help)\n", tool, kv.first.c_str());
            return false;
        }
    }
    return true;
}

inline void lensFromParams(const Params& p, ca::WavefrontParams& w, ca::ApertureParams& a, ca::ImageGeometry& g, ca::PsfOptions& opt)
{
    w.fNumber = num(p, "fnumber", 2.8);
    w.lambda0Um = num(p, "lambda0", 550.) * 1e-3;
    w.W040 = num(p, "w040", 0.);
    w.W131 = num(p, "w131", 0.);
    w.W222 = num(p, "w222", 0.);
    w.W220 = num(p, "w220", 0.);
    w.W311 = num(p, "w311", 0.);
    w.W020 = num(p, "w020", 0.);
    w.axialCA = num(p, "axial_ca", 0.);
    w.lateralCA = num(p, "lateral_ca", 0.);
    w.spherochromatism = num(p, "spherochromatism", 0.);
    w.zernike.clear();
    std::stringstream ss( str(p, "zernike", "") );
    std::string item;
    while ( std::getline(ss, item, ',') ) {
        const size_t c = item.find(':');
        if (c == std::string::npos) {
            continue;
        }
        const int j = std::atoi( item.substr(0, c).c_str() );
        if (j < 4) {
            std::fprintf(stderr, "zernike index %d ignored (piston and tilts are not aberrations)\n", j);
            continue;
        }
        w.zernike.push_back( std::make_pair( j, std::atof( item.substr(c + 1).c_str() ) ) );
    }
    a.blades = (int)num(p, "blades", 0);
    a.roundness = num(p, "roundness", 0.);
    a.rotationDeg = num(p, "rotation", 0.);
    a.aspect = num(p, "aspect", 1.);
    a.vignetting = num(p, "vignetting", 0.);
    a.vignettingStart = num(p, "vignetting_start", 0.5);
    g.filmbackWidthMm = num(p, "filmback", 24.89);
    g.pixelAspect = num(p, "pixel_aspect", 1.);
    g.centerX = num(p, "center_x", 0.5);
    g.centerY = num(p, "center_y", 0.5);
    const std::string m = str(p, "method", "auto");
    opt.method = m == "wave" ? ca::eMethodWave : (m == "geometric" ? ca::eMethodGeometric : ca::eMethodAuto);
    opt.maxFFT = (int)num(p, "max_fft", 1024);
}

// Lens model from the parameters: a real lens (key lens = built-in name or .lens file) or the
// parametric lens. Returns null (and prints the error) on failure.
inline std::unique_ptr<ca::LensModel> lensModelFromParams(const char* tool, const Params& p, const ca::WavefrontParams& w,
                                                          const ca::ApertureParams& a)
{
    const std::string lens = str(p, "lens", "");
    if ( lens.empty() ) {
        return std::unique_ptr<ca::LensModel>( new ca::ParametricLensModel(w, a) );
    }
    ca::LensPrescription pr;
    std::string err;
    bool ok = (lens.find(".lens") != std::string::npos) ? ca::loadLensFile(lens, pr, &err) : ca::loadBuiltinLens(lens, pr, &err);
    if (!ok) {
        std::fprintf(stderr, "%s: %s\n", tool, err.c_str());
        return std::unique_ptr<ca::LensModel>();
    }
    ca::RealLensSettings s;
    s.focalMm = num(p, "focal", 50.);
    s.fNumber = p.count("fnumber") ? num(p, "fnumber", 0.) : 0.;
    s.focusDistanceMm = num(p, "focus_distance", 0.) * 1000.;
    s.focusOffsetMm = num(p, "focus_offset", 0.);
    s.bestFocus = num(p, "design_focus", 0.) == 0.;
    s.blades = a.blades;
    s.roundness = a.roundness;
    s.rotationDeg = a.rotationDeg;
    s.aberrationGain = num(p, "aberration_gain", 1.);
    s.distortionGain = num(p, "distortion_gain", 1.);
    s.lateralColourGain = num(p, "lateral_gain", 1.);
    s.defocusWaves = w.W020;
    s.unvignettedField = num(p, "unvignetted_field", 0.6);
    s.cornerPupil = num(p, "corner_pupil", 0.6);
    ca::RealLens L;
    if ( !L.setup(pr, s, &err) ) {
        std::fprintf(stderr, "%s: %s: %s\n", tool, lens.c_str(), err.c_str());
        return std::unique_ptr<ca::LensModel>();
    }
    std::printf("Real lens: %s, focal %.2f mm, f/%.2f (design f/%.2f), working f/%.2f\n", pr.name.c_str(), L.efl(), L.fNumber(),
                pr.designFNumber, L.workingFNumber());
    return std::unique_ptr<ca::LensModel>( new ca::RealLensModel(L) );
}

static const char* const kLensHelp =
    "Lens (aberration coefficients in waves at lambda0, at the pupil edge and image corner):\n"
    "  fnumber=2.8  lambda0=550 (nm)\n"
    "  w040 (spherical)  w131 (coma)  w222 (astigmatism)  w220 (field curvature)\n"
    "  w311 (distortion, <0 barrel)  w020 (defocus)\n"
    "  axial_ca  lateral_ca  spherochromatism\n"
    "  zernike=j:coef,j:coef  (Noll index >= 4, RMS waves)\n"
    "Aperture:\n"
    "  blades=0 (0 = round)  roundness=0  rotation=0 (deg)  aspect=1 (height/width)\n"
    "  vignetting=0 (mechanical vignetting: how deep the barrel cuts into the pupil in the corner,\n"
    "               in pupil radii; 0.5 = cat-eye bokeh)  vignetting_start=0.5 (field where it begins)\n"
    "Sensor:\n"
    "  filmback=24.89 (mm, image width on the sensor)  center_x=0.5 center_y=0.5 (optical centre)\n"
    "  pixel_aspect=1 (2 = anamorphic 2x plate; the lens is modelled in the desqueezed space)\n"
    "Real lens (instead of the coefficients above):\n"
    "  lens=double_gauss_f2|wide_angle_100|telephoto_f56|cooke_triplet_f5|petzval_f14 or a .lens file\n"
    "  focal=50 (mm, the lens is scaled to it)  fnumber (default: the design aperture)\n"
    "  focus_distance=0 (m, 0 = infinity)  focus_offset=0 (mm)  design_focus=0 (1: image plane of the file)\n"
    "  w020 = extra defocus (waves)  aberration_gain=1  distortion_gain=1  lateral_gain=1\n"
    "  unvignetted_field=0.6 corner_pupil=0.6 (automatic barrel sizes)\n"
    "PSF computation:\n"
    "  method=auto|wave|geometric  max_fft=1024 (larger wave-optics FFTs switch to the hybrid method)\n";

} // namespace catool
