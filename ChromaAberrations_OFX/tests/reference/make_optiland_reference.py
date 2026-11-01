#!/usr/bin/env python3
# ChromaAberrations_OFX - reference data for test_lens, computed with Optiland (MIT licence,
# https://github.com/HarrisonKramer/optiland), an independent ray tracer.
#   PYTHONPATH=<optiland checkout> python3 make_optiland_reference.py > optiland_reference.csv
# Real ray intercepts on the image plane for each built-in lens, at 3 wavelengths, 4 field
# angles and 5 pupil heights, plus the paraxial focal length:
#  - "chief": chief ray, aimed iteratively at the centre of the stop (distortion, lateral colour);
#  - "y": rays through points of the paraxial entrance pupil (no aiming: pure tracing check;
#    the aiming conventions of the two tracers size the stop differently at the rim).
import numpy as np
from optiland.samples.objectives import ObjectiveUS008879901, WideAngle100FOV, Telephoto, CookeTriplet, PetzvalLens

# (lens file, Optiland sample, unit of the Optiland sample relative to the lens file)
LENSES = [("double_gauss_f2", ObjectiveUS008879901, 1.), ("wide_angle_100", WideAngle100FOV, 1.),
          ("telephoto_f56", Telephoto, 25.4), ("cooke_triplet_f5", CookeTriplet, 1.), ("petzval_f14", PetzvalLens, 25.4)]
print("lens,kind,field,pupil,wavelength_um,value")
for name, cls, unit in LENSES:
    l = cls()
    print("%s,unit,0,0,0,%.10g" % (name, unit))
    # paraxial focal length at the primary wavelength of the sample
    print("%s,efl,0,0,%.7f,%.10g" % (name, float(l.primary_wavelength), float(l.paraxial.f2())))
    hmax = max(float(f.y) for f in l.fields.fields)
    for mode in ("iterative", "paraxial"):
        l.ray_tracer.set_aiming(mode)
        for fr in (0.0, 0.5, 0.7, 1.0):
            for wl in (0.4861327, 0.5875618, 0.6562725):
                for py in ((0.0,) if mode == "iterative" else (-1.0, -0.7, 0.0, 0.7, 1.0)):
                    rays = l.trace_generic(Hx=0.0, Hy=fr, Px=0.0, Py=py, wavelength=wl)
                    y = float(np.ravel(rays.y)[0])
                    print("%s,%s,%.3f,%.2f,%.7f,%.12g" % (name, "chief" if mode == "iterative" else "y", fr * hmax, py, wl, y))
