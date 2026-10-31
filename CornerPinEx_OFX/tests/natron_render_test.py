# End-to-end test of CornerPinEx_OFX inside a real OFX host (Natron, headless):
#   CPGPU_OUT=<outdir> OFX_PLUGIN_PATH=<build dir> NatronRenderer -t natron_render_test.py
# Renders the same corner pin with Natron's CornerPin and with CornerPinEx_OFX, so
# that the images can be compared (compare_renders.py).
import sys
import NatronEngine

app = app1
import os
outdir = os.environ.get("CPGPU_OUT", "/tmp/cpgpu_out")

# source: checkerboard + text-like detail, 1920x1080
src = app.createNode("net.sf.openfx.CheckerBoardPlugin")
src.getParam("boxSize").setValue(24, 0)
src.getParam("boxSize").setValue(24, 1)

TO = [(420, 180), (1650, 300), (1500, 900), (300, 1000)]
OFFSETS = [(30, -20), (0, 0), (-45, 15), (10, 40)]
FROM = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]
FROM_OFFSETS = [(50, 30), (-20, 0), (0, -60), (25, 10)]

FILTERS = {"cubic": 3, "keys": 4, "ewagaussian": 10, "ewalanczos3": 11}


def make_pin(plugin_id, to, filt, offsets=None, motion=False, use_gpu=True, clamp=False, frm=None, from_offsets=None):
    pin = app.createNode(plugin_id)
    pin.connectInput(0, src)
    for i, (x, y) in enumerate(to):
        p = pin.getParam("to%d" % (i + 1))
        if motion:
            # animate the corners between frames 1 and 2: the shutter at frame 1 sees this motion
            p.setValueAtTime(x, 1, 0)
            p.setValueAtTime(y, 1, 1)
            p.setValueAtTime(x + 60, 2, 0)
            p.setValueAtTime(y + 25, 2, 1)
        else:
            p.setValue(x, 0)
            p.setValue(y, 1)
    if frm is not None:
        for i, (x, y) in enumerate(frm):
            pin.getParam("from%d" % (i + 1)).setValue(x, 0)
            pin.getParam("from%d" % (i + 1)).setValue(y, 1)
    pin.getParam("filter").setValue(filt)
    pin.getParam("clamp").setValue(clamp)
    if motion:
        pin.getParam("motionBlur").setValue(1.0)
        pin.getParam("shutter").setValue(1.0)
        pin.getParam("shutterOffset").setValue(1)  # start
    if offsets is not None:
        for i, (x, y) in enumerate(offsets):
            pin.getParam("to%dOffsetX" % (i + 1)).setValue(x)
            pin.getParam("to%dOffsetY" % (i + 1)).setValue(y)
    if from_offsets is not None:
        for i, (x, y) in enumerate(from_offsets):
            pin.getParam("from%dOffsetX" % (i + 1)).setValue(x)
            pin.getParam("from%dOffsetY" % (i + 1)).setValue(y)
    if plugin_id.startswith("community") and not use_gpu:
        pin.getParam("useGPU").setValue(False)
    return pin


def write(node, name):
    w = app.createNode("fr.inria.openfx.WriteOIIO")
    w.connectInput(0, node)
    w.getParam("filename").set("%s/%s.png" % (outdir, name))
    app.render(w, 1, 1)
    print("rendered", name)


cases = [
    ("natron_cubic", make_pin("net.sf.openfx.CornerPinPlugin", TO, 3)),
    ("gpu_cubic_cpu", make_pin("community.openfx.CornerPinEx_OFX", TO, 3, use_gpu=False)),
    ("gpu_cubic", make_pin("community.openfx.CornerPinEx_OFX", TO, 3)),
    ("gpu_ewagaussian", make_pin("community.openfx.CornerPinEx_OFX", TO, 10)),
    ("gpu_ewalanczos3", make_pin("community.openfx.CornerPinEx_OFX", TO, 11)),
    ("gpu_ewalanczos3_clamp", make_pin("community.openfx.CornerPinEx_OFX", TO, 11, clamp=True)),
    # offsets: CornerPinEx_OFX with offsets == Natron CornerPin with the summed points
    ("natron_offset", make_pin("net.sf.openfx.CornerPinPlugin",
                               [(t[0] + o[0], t[1] + o[1]) for t, o in zip(TO, OFFSETS)], 4)),
    ("gpu_offset", make_pin("community.openfx.CornerPinEx_OFX", TO, 4, offsets=OFFSETS)),
    # "from" offsets == Natron CornerPin with the summed "from" points
    ("natron_fromoffset", make_pin("net.sf.openfx.CornerPinPlugin", TO, 4,
                                   frm=[(f[0] + o[0], f[1] + o[1]) for f, o in zip(FROM, FROM_OFFSETS)])),
    ("gpu_fromoffset", make_pin("community.openfx.CornerPinEx_OFX", TO, 4, frm=FROM, from_offsets=FROM_OFFSETS)),
    # motion blur
    ("natron_mb", make_pin("net.sf.openfx.CornerPinPlugin", TO, 3, motion=True)),
    ("gpu_mb", make_pin("community.openfx.CornerPinEx_OFX", TO, 3, motion=True)),
    ("gpu_mb_ewalanczos3", make_pin("community.openfx.CornerPinEx_OFX", TO, 11, motion=True)),
]
for name, node in cases:
    write(node, name)

st = cases[2][1].getParam("gpuStatus")
print("gpuStatus:", st.get() if st else None)
app.quit() if hasattr(app, "quit") else None
quit()
