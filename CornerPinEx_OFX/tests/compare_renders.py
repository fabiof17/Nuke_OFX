# Compare the images written by natron_render_test.py
import sys
import numpy as np
from PIL import Image

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cpgpu_out"
def load(n):
    return np.asarray(Image.open("%s/%s.png" % (d, n)).convert("RGBA")).astype(np.int32)

pairs = [("natron_cubic", "gpu_cubic_cpu"), ("natron_cubic", "gpu_cubic"),
         ("natron_offset", "gpu_offset"), ("natron_fromoffset", "gpu_fromoffset"), ("natron_mb", "gpu_mb"),
         ("gpu_cubic", "gpu_ewagaussian"), ("gpu_cubic", "gpu_ewalanczos3"), ("gpu_mb", "gpu_mb_ewalanczos3")]
for a, b in pairs:
    A, B = load(a), load(b)
    diff = np.abs(A - B)
    print("%-15s vs %-20s max diff %3d  mean diff %.4f  pixels differing %.3f%%" %
          (a, b, diff.max(), diff.mean(), 100.0 * (diff.max(axis=2) > 0).mean()))
