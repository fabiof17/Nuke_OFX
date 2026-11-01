// ChromaAberrations_OFX - optical prescriptions (.lens files)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "lensfile.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ca {

namespace {

struct Builtin
{
    const char* id;
    const char* text;
};

const Builtin kBuiltins[] = {
#include "lenses_builtin.inc"
};

std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::string lower(std::string s)
{
    for (char& c : s) c = (char)std::tolower( (unsigned char)c );
    return s;
}

bool number(const std::string& t, double& v)
{
    const std::string l = lower(t);
    if ( (l == "inf") || (l == "infinity") || (l == "flat") ) {
        v = 0.;   // flat surface (curvature 0)
        return true;
    }
    char* e = 0;
    v = std::strtod(t.c_str(), &e);
    return e && (*e == 0) && !t.empty();
}

} // namespace

bool parseLensText(const std::string& text, LensPrescription& lens, std::string* error)
{
    lens = LensPrescription();
    std::istringstream in(text);
    std::string line;
    int n = 0;
    bool inSurfaces = false;
    auto fail = [&](const std::string& msg) {
        if (error) {
            std::ostringstream o;
            o << "line " << n << ": " << msg;
            *error = o.str();
        }
        return false;
    };
    while ( std::getline(in, line) ) {
        ++n;
        line = trim( line.substr( 0, line.find('#') ) );
        if ( line.empty() ) {
            continue;
        }
        if (!inSurfaces) {
            if (lower(line) == "surfaces") {
                inSurfaces = true;
                continue;
            }
            const size_t eq = line.find('=');
            if (eq == std::string::npos) {
                return fail("expected key = value or 'surfaces'");
            }
            const std::string k = lower( trim( line.substr(0, eq) ) ), v = trim( line.substr(eq + 1) );
            if (k == "name") {
                lens.name = v;
            } else if (k == "source") {
                lens.source = v;
            } else if (k == "fnumber") {
                lens.designFNumber = std::atof( v.c_str() );
            } else if (k == "half_field") {
                lens.designHalfFieldDeg = std::atof( v.c_str() );
            }   // other keys (reference_focal, ...) are informative
            continue;
        }
        std::istringstream ls(line);
        std::vector<std::string> tok;
        std::string t;
        while (ls >> t) {
            tok.push_back(t);
        }
        if (tok.size() < 3) {
            return fail("a surface needs: radius thickness material");
        }
        LensSurface s;
        double r;
        if ( !number(tok[0], r) ) {
            return fail("bad radius '" + tok[0] + "'");
        }
        s.radius = r;
        if ( !number(tok[1], s.thickness) ) {
            return fail("bad thickness '" + tok[1] + "'");
        }
        std::string gerr;
        if ( !findGlass(tok[2], s.glass, &gerr) ) {
            return fail(gerr);
        }
        for (size_t i = 3; i < tok.size(); ++i) {
            const std::string w = lower(tok[i]);
            double v;
            if (w == "stop") {
                s.stop = true;
            } else if (w == "auto") {
                s.semiDiameter = 0.;
            } else if (w.compare(0, 2, "k=") == 0) {
                s.conic = std::atof( w.c_str() + 2 );
            } else if ( (w[0] == 'a') && (w.find('=') != std::string::npos) ) {
                const int order = std::atoi( w.c_str() + 1 );
                if ( (order < 4) || (order % 2) || (order > 30) ) {
                    return fail("aspheric terms are a4, a6, a8, ...");
                }
                const size_t idx = (order - 4) / 2;
                if (s.asph.size() <= idx) {
                    s.asph.resize(idx + 1, 0.);
                }
                s.asph[idx] = std::atof( w.c_str() + w.find('=') + 1 );
            } else if ( number(tok[i], v) ) {
                s.semiDiameter = std::fabs(v);
            } else {
                return fail("unknown word '" + tok[i] + "'");
            }
        }
        if (s.stop) {
            if (lens.stopIndex >= 0) {
                return fail("only one stop");
            }
            lens.stopIndex = (int)lens.surfaces.size();
        }
        lens.surfaces.push_back(s);
    }
    if ( lens.surfaces.empty() ) {
        if (error) *error = "no surfaces";
        return false;
    }
    if (lens.stopIndex < 0) {
        if (error) *error = "no aperture stop (add 'stop' on one surface line)";
        return false;
    }
    if (lens.designFNumber <= 0.) {
        if (error) *error = "missing 'fnumber = ...'";
        return false;
    }
    if (lens.designHalfFieldDeg <= 0.) {
        lens.designHalfFieldDeg = 20.;
    }
    if ( lens.name.empty() ) {
        lens.name = "lens";
    }
    return true;
}

bool loadLensFile(const std::string& path, LensPrescription& lens, std::string* error)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) {
        if (error) *error = "cannot read " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    if ( !parseLensText(ss.str(), lens, error) ) {
        if (error) *error = path + ": " + *error;
        return false;
    }
    return true;
}

int builtinLensCount()
{
    return (int)( sizeof(kBuiltins) / sizeof(kBuiltins[0]) );
}

const char* builtinLensId(int i)
{
    return (i >= 0 && i < builtinLensCount()) ? kBuiltins[i].id : "";
}

const char* builtinLensText(int i)
{
    return (i >= 0 && i < builtinLensCount()) ? kBuiltins[i].text : "";
}

bool loadBuiltinLens(const std::string& id, LensPrescription& lens, std::string* error)
{
    for (int i = 0; i < builtinLensCount(); ++i) {
        if (id == kBuiltins[i].id) {
            return parseLensText(kBuiltins[i].text, lens, error);
        }
    }
    if (error) *error = "unknown built-in lens " + id;
    return false;
}

} // namespace ca
