// ChromaAberrations_OFX - minimal OpenEXR writer (scanline, uncompressed, 32-bit float)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <string>
#include <vector>

namespace ca {

struct ExrChannel
{
    std::string name;            // e.g. "R", "G", "B", "lambda.450nm.Y"
    const float* data;           // width * height values, row-major, first row = top of the image
};

// Writes an uncompressed float EXR. Returns false (and fills *err) on failure.
bool writeExr(const std::string& path, int width, int height, const std::vector<ExrChannel>& channels, std::string* err);

} // namespace ca
