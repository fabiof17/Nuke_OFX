// ChromaAberrations_OFX - real lenses: sequential ray tracing through an optical prescription
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// The lens is traced in double precision, wavelength by wavelength (glass dispersion), with
// real ray aiming at the aperture stop. For a field point it gives:
//  - the chief ray landing on the sensor (distortion, lateral colour);
//  - the wavefront: optical path difference to the reference sphere centred on the chief ray
//    image point, fitted with Zernike polynomials over the exit pupil;
//  - the pupil actually open (rays clipped by the lens barrels: mechanical vignetting);
//  - the relative illumination.
// The lens is rotationally symmetric: these are computed along one field radius and rotated.
//
// Axes: z along the optical axis, from the object; the first surface vertex is at z = 0.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lensfile.h"
#include "lensmodel.h"

namespace ca {

struct Vec3
{
    double x = 0., y = 0., z = 0.;
    Vec3() {}
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
};

struct RealLensSettings
{
    double focalMm = 0.;         // scale the lens to this focal length (0: as in the file)
    double fNumber = 0.;         // aperture (0: design f-number; never wider than the design)
    double focusDistanceMm = 0.; // object distance from the first vertex (0: infinity)
    double focusOffsetMm = 0.;   // extra sensor displacement (positive: away from the lens)
    bool bestFocus = true;       // sensor at the best focus on axis (smallest RMS wavefront at 587.6 nm);
                                 // false: at the image distance of the prescription
    // automatic semi-diameters: the full aperture passes up to this fraction of the design field,
    // then the open part of the pupil shrinks to 'cornerPupil' (radius fraction) at the full field
    double unvignettedField = 0.6;
    double cornerPupil = 0.6;
    // diaphragm shape (at the stop)
    int blades = 0;
    double roundness = 0.;
    double rotationDeg = 0.;
    // artistic gains (1 = the real lens)
    double aberrationGain = 1.;  // wavefront (all aberrations)
    double distortionGain = 1.;  // chief ray displacement at the reference wavelength
    double lateralColourGain = 1.; // chief ray displacement relative to the reference wavelength
    double defocusWaves = 0.;    // added defocus (waves at 550 nm, at the pupil edge)
};

class RealLens
{
public:
    // Scales the lens, computes the paraxial data, the stop size and the automatic semi-diameters.
    bool setup(const LensPrescription& lens, const RealLensSettings& settings, std::string* error = 0);

    const LensPrescription& prescription() const { return _lens; }
    const RealLensSettings& settings() const { return _set; }

    // paraxial data at the reference wavelength (587.56 nm), lens units after scaling (mm)
    double efl() const { return _efl; }
    double eflAt(double nm) const;                     // paraxial focal length at another wavelength
    double bfl() const { return _bfl; }
    double entrancePupilZ() const { return _epZ; }
    double entrancePupilRadius() const { return _epR; }
    double exitPupilZ() const { return _xpZ; }
    double stopRadius() const { return _stopR; }
    double imageZ() const { return _zImage; }
    double workingFNumber() const { return _workingN; }
    double fNumber() const { return _fNumber; }      // the aperture used (clamped to the design)
    double fieldScale() const { return _k; }          // ideal image height per unit field (tan or object height)
    bool infiniteObject() const { return _infinite; }
    double semiDiameter(int i) const { return _sd[i]; }
    int surfaceCount() const { return (int)_lens.surfaces.size(); }

    struct Ray
    {
        Vec3 p, d;          // position, unit direction
        double opl = 0.;    // optical path length so far (lens units)
    };

    // Field point given by the ideal image position on the sensor (X, Y), lens units
    struct Field
    {
        double tx = 0., ty = 0.;    // infinite object: direction tangents
        Vec3 object;                // finite object: object point
    };
    Field fieldAt(double X, double Y) const;

    // Ray from field f, with aiming variables v (start point or direction tangents)
    Ray makeRay(const Field& f, double v0, double v1) const;
    // Traces up to surface 'upTo' (included; -1 = all, then to the image plane).
    // clip: stop at the semi-diameters and the stop radius. Returns false if blocked or missed.
    bool trace(Ray& r, double nm, bool clip, int upTo = -1, int* blockedAt = 0) const;
    // Aiming variables of the ray through point (sx, sy) of the stop (lens units). 'v' holds the
    // initial guess on input (if guess is true). Returns false if it does not converge.
    bool aim(const Field& f, double sx, double sy, double nm, double v[2], bool guess = false) const;
    // Ray through point (ex, ey) of the paraxial entrance pupil plane (no aiming)
    Ray rayThroughEntrancePupil(const Field& f, double ex, double ey) const;
    // Chief ray landing on the sensor (lens units); false if the chief ray cannot be traced
    bool chiefLanding(double X, double Y, double nm, double& x, double& y) const;

    double index(int surface, double nm) const;   // medium after the surface

private:
    bool intersect(int i, Ray& r) const;
    void normalAt(int i, double x, double y, Vec3& n) const;
    double sag(int i, double r) const;
    double sagSlope(int i, double r) const;
    bool paraxial(double nm, double y0, double u0, double z0, std::vector<double>& y, std::vector<double>& u) const;
    double onAxisRmsOpd(double zImage) const;      // RMS wavefront on axis with the sensor at zImage (lens units)
    void initialGuess(const Field& f, double sx, double sy, double v[2]) const;

    LensPrescription _lens;
    RealLensSettings _set;
    std::vector<double> _z, _c, _sd;   // vertex positions, curvatures, semi-diameters
    double _efl = 0., _bfl = 0., _epZ = 0., _epR = 0., _xpZ = 0., _stopR = 0., _zImage = 0.;
    double _workingN = 0., _fNumber = 0., _k = 1., _stopGain = 1., _zStart = 0., _zObject = 0.;
    bool _infinite = true;
};

// The real lens as a lens model for the renderer (see lensmodel.h)
class RealLensModel : public LensModel
{
public:
    explicit RealLensModel(const RealLens& lens) : _lens(lens) {}

    bool prepare(const ImageGeometry& g, const std::vector<double>& nm,
                 std::function<void(int, int)> progress, std::string* error) override;
    void psfInputs(double hx, double hy, int l, WavefrontParams& w, ApertureParams& a) const override;
    void displacementUm(double hx, double hy, int l, double& dx, double& dy) const override;
    double illumination(double hx, double hy) const override;
    bool includesNaturalFalloff() const override { return true; }
    bool cheapIllumination() const override { return true; }

    const RealLens& lens() const { return _lens; }
    double maxFieldMm() const { return _rmax; }      // sensor radius covered (corner)
    double imageHalfFieldDeg() const;               // half field of view of the image corner
    int fitTerms() const { return kTerms; }
    int tracedWavelengths() const { return _tracedWavelengths; }   // the others are interpolated
    void setMaxTracedWavelengths(int n) { _maxTraced = n; }

    // Wavefront at one field radius (along +y) and wavelength: Zernike coefficients (Noll j = 1..J,
    // micrometres, exit pupil coordinates q), effective f-number, pupil map q -> stop. For tests.
    struct FieldWavefront
    {
        bool ok = false;
        std::vector<double> coef;   // [J + 1], index j
        double fNumber = 0.;
        double pupil[4] = { 1., 0., 0., 1. };  // q = pupil * p (p: stop coordinates / stop radius)
        double rmsUm = 0.;          // RMS of the OPD over the open pupil, tilt and piston removed
        double fitResidualUm = 0.;  // RMS of the fit residual
        double gradientCheck = 0.;  // regression slope of the traced transverse aberrations on -2N grad W (1 = consistent)
        double illum = 0.;          // relative illumination factor (not normalised)
        std::vector<float> mask;    // vignetting mask over q (kMask x kMask), 1 = open
    };
    FieldWavefront fieldWavefront(double r, double nm, bool withMask) const;

    static const int kTerms = 45;   // Zernike terms (radial order <= 8)
    static const int kMask = 65;
    static const int kGrid = 97;    // OPD grid given to the PSF engine

private:
    const RealLens _lens;
    double _rmax = 1.;
    double _pitchUm = 1.;
    std::vector<double> _nm;
    // radial tables
    std::vector<double> _rPsf;                       // field radii of the wavefront samples
    std::vector<std::vector<FieldWavefront> > _wf;   // [radius][wavelength]
    std::vector<double> _rDist;                      // field radii of the chief ray table
    std::vector<std::vector<double> > _dist;         // [wavelength][radius]: landing - ideal (lens units)
    std::vector<double> _illum;                      // [radius of _rPsf]: relative illumination
    int _refL = 0;                                   // wavelength index closest to 587.6 nm
    int _tracedWavelengths = 0;
    int _maxTraced = 9;
    std::shared_ptr<std::vector<float> > _basis;     // Zernike values on the OPD grid [j][node]
};

} // namespace ca
