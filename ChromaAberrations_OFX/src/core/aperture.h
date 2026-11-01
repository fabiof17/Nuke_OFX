// ChromaAberrations_OFX - aperture (diaphragm) shape
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <memory>
#include <vector>

namespace ca {

// Shape of the diaphragm in normalised pupil coordinates: the pupil fits in the unit
// circle (the f-number is defined by that circle).
struct ApertureParams
{
    int blades = 0;           // 0 = perfectly round; otherwise number of straight blades (>= 3)
    double roundness = 0.;    // 0 = straight blades (polygon), 1 = round
    double rotationDeg = 0.;  // rotation of the polygon
    double aspect = 1.;       // height / width of the pupil (> 1: tall oval, anamorphic look)

    // Mechanical vignetting: off axis, the lens barrel (front or rear element) clips the pupil,
    // giving cat-eye bokeh and darker corners. Modelled as a circle a bit larger than the pupil,
    // that moves across it with the field position H (normalised, 1 in the image corner).
    // It starts clipping at |H| = vignettingStart; 'vignetting' is how deep it cuts into the
    // pupil in the corner, in pupil radii (0 = none, 0.5 = cat-eye, 1 = half the pupil gone).
    double vignetting = 0.;
    double vignettingStart = 0.5;
    double fieldX = 0., fieldY = 0.;  // field point the pupil is seen from (set by computePsf)

    // Traced pupil (real lenses). The pupil coordinates q are those of the exit pupil; the
    // diaphragm (unit circle and blades) is tested at p = pupilMap q, its own coordinates.
    // 'mask' holds maskN x maskN transmission values over q in [-1, 1]^2 (row-major, y up):
    // a point is open where the bilinear value is >= 0.5 (rays clipped by the other elements).
    std::shared_ptr<const std::vector<float> > mask;
    int maskN = 0;
    bool hasPupilMap = false;
    double pupilMap[4] = { 1., 0., 0., 1. };
};

// True if the pupil point (x, y) lets light through.
bool apertureInside(const ApertureParams& a, double x, double y);

// Fraction of the square cell of side 'cell' centred on (x, y) that is open
// (4x4 supersampling near the edge, for an anti-aliased pupil).
double apertureCoverage(const ApertureParams& a, double x, double y, double cell);

// Radius and shift per unit of field of the clipping circle (pupil radius = 1)
void vignettingCircle(const ApertureParams& a, double& radius, double& shiftPerField);

// Open area of the pupil seen from the field point (fieldX, fieldY), relative to the full
// unclipped diaphragm: the mechanical part of the relative illumination.
double apertureTransmission(const ApertureParams& a, int samples = 256);

} // namespace ca
