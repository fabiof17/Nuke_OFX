// ChromaAberrations_OFX - camera spectral sensitivities
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "camera.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ca {

#include "cameras.inc"

double CameraSensitivity::at(int c, double l) const
{
    if ( !valid() || (l < nm.front()) || (l > nm.back()) ) {
        return 0.;
    }
    const size_t k = std::min( (size_t)(std::upper_bound(nm.begin(), nm.end(), l) - nm.begin()), nm.size() - 1 );
    const size_t j = k - 1;
    const double f = (nm[k] > nm[j]) ? (l - nm[j]) / (nm[k] - nm[j]) : 0.;
    return s[c][j] + f * (s[c][k] - s[c][j]);
}

int builtinCameraCount() { return kCameraCount; }
const char* builtinCameraId(int i) { return (i >= 0 && i < kCameraCount) ? kCameras[i].id : ""; }
const char* builtinCameraName(int i) { return (i >= 0 && i < kCameraCount) ? kCameras[i].name : ""; }

bool builtinCamera(const std::string& id, CameraSensitivity& cam)
{
    for (int i = 0; i < kCameraCount; ++i) {
        if (id == kCameras[i].id) {
            cam = CameraSensitivity();
            cam.id = kCameras[i].id;
            cam.name = kCameras[i].name;
            for (int k = 0; k < kCameraSamples; ++k) {
                cam.nm.push_back(380. + 5. * k);
                for (int c = 0; c < 3; ++c) cam.s[c].push_back(kCameras[i].s[c][k]);
            }
            return true;
        }
    }
    return false;
}

int trainingPatchCount() { return kTrainingCount; }
const float* trainingPatch(int i) { return kTraining[i]; }

static bool finish(CameraSensitivity& cam, std::vector<std::pair<double, std::vector<double> > >& rows, std::string* error)
{
    std::sort(rows.begin(), rows.end(), [](const std::pair<double, std::vector<double> >& a,
                                            const std::pair<double, std::vector<double> >& b) { return a.first < b.first; });
    cam.nm.clear();
    for (int c = 0; c < 3; ++c) cam.s[c].clear();
    double top[3] = { 0., 0., 0. };
    for (const auto& r : rows) {
        if ( !cam.nm.empty() && (r.first <= cam.nm.back()) ) continue;
        cam.nm.push_back(r.first);
        for (int c = 0; c < 3; ++c) {
            cam.s[c].push_back(r.second[c]);
            top[c] = std::max(top[c], r.second[c]);
        }
    }
    if ( (cam.nm.size() < 5) || (top[0] <= 0.) || (top[1] <= 0.) || (top[2] <= 0.) ) {
        if (error) *error = "not a camera sensitivity file (needs lines \"nm r g b\" with positive values)";
        return false;
    }
    if ( (cam.nm.front() > 420.) || (cam.nm.back() < 680.) ) {
        if (error) *error = "the sensitivities must cover at least 420-680 nm";
        return false;
    }
    for (int c = 0; c < 3; ++c) {
        for (double& v : cam.s[c]) v /= top[c];
    }
    return true;
}

bool loadCameraFile(const std::string& path, CameraSensitivity& cam, std::string* error)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    cam = CameraSensitivity();
    const size_t slash = path.find_last_of("/\\");
    cam.id = cam.name = slash == std::string::npos ? path : path.substr(slash + 1);
    std::vector<std::pair<double, std::vector<double> > > rows;
    if (text.find('{') != std::string::npos) {
        // rawtoaces-data JSON: "380": [r, g, b], ...
        size_t p = text.find("\"main\"");
        p = text.find("\"main\"", p + 1);   // the data block comes after the index block
        if (p == std::string::npos) p = text.find("\"data\"");
        while ( p != std::string::npos && (p = text.find('"', p + 1)) != std::string::npos ) {
            char* e = 0;
            const double nm = std::strtod(text.c_str() + p + 1, &e);
            if ( (e == text.c_str() + p + 1) || (*e != '"') ) continue;
            size_t b = text.find('[', p);
            if (b == std::string::npos) break;
            std::vector<double> v;
            const char* q = text.c_str() + b + 1;
            for (int c = 0; c < 3; ++c) {
                char* qe = 0;
                v.push_back(std::strtod(q, &qe));
                q = qe;
                while (*q == ',' || *q == ' ' || *q == '\n' || *q == '\r' || *q == '\t') ++q;
            }
            rows.push_back(std::make_pair(nm, v));
            p = text.find(']', b);
        }
    } else {
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            for (char& ch : line) if (ch == ',' || ch == ';' || ch == '\t') ch = ' ';
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            double nm, r, g, b;
            if (ls >> nm >> r >> g >> b) rows.push_back(std::make_pair(nm, std::vector<double>{ r, g, b }));
        }
    }
    return finish(cam, rows, error);
}

} // namespace ca
