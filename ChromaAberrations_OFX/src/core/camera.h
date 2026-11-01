// ChromaAberrations_OFX - camera spectral sensitivities
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// By default the renderer turns light back into RGB with the CIE 1931 observer. A camera sees
// the colours of the lens fringes through its own red, green and blue filters instead: with a
// camera selected, the plate's colours are treated as that camera's RGB, carried into the
// working space by a 3 x 3 matrix (the camera's input transform, fitted here on 190 reflectance
// spectra, white preserved). Colours without lens effects come out unchanged either way; only the
// colours of the separated wavelengths (fringes, rainbow edges of highlights) change.
//
// Built-in data: rawtoaces-data of the Academy Software Foundation (Apache-2.0), 52 cameras.

#pragma once

#include <string>
#include <vector>

namespace ca {

struct CameraSensitivity
{
    std::string id, name;
    std::vector<double> nm;      // increasing wavelengths
    std::vector<double> s[3];    // R, G, B sensitivities at nm (any scale)

    bool valid() const { return nm.size() >= 2; }
    double at(int c, double lambdaNm) const;   // linear interpolation, 0 outside
};

int builtinCameraCount();
const char* builtinCameraId(int i);
const char* builtinCameraName(int i);
bool builtinCamera(const std::string& id, CameraSensitivity& cam);

// Text file: one line per wavelength "nm r g b" (spaces, tabs, commas or semicolons; lines
// starting with # and a header line are ignored), or a rawtoaces-data JSON file.
bool loadCameraFile(const std::string& path, CameraSensitivity& cam, std::string* error);

// Training reflectances (190 patches, 380-780 nm, 5 nm steps) used to fit the camera matrix
int trainingPatchCount();
const float* trainingPatch(int i);   // 81 values

} // namespace ca
