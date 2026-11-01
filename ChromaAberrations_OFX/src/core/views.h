// ChromaAberrations_OFX - test chart and PSF map views (plugin and tools)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <string>

#include "render.h"

namespace ca {

// Test images in linear light: "siemens" (Siemens stars on mid grey), "points" (HDR point
// lights, white and saturated colours, on a regular grid) or "chart" (both).
void makeTestChart(const std::string& kind, int width, int height, ImageRGB& img);

// PSF map: the PSFs of the plan (white light, summed over the wavelengths with the channel
// responses) drawn at their grid positions over a dimmed background, each normalised to a
// peak of 1 and shown with a 2.2 gamma (it is a diagnostic view, not linear light),
// enlarged 'zoom' times (0 = automatic: the largest PSF fills most of a grid
// cell). 'background' may be empty (black).
void makePsfMapView(const LensPlan& plan, const ImageRGB* background, int zoom, ImageRGB& out);

} // namespace ca
