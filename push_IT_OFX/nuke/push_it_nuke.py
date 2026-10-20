# push_it_nuke.py
# Nuke integration for the push_IT_OFX OFX plug-in: node colour in the Node Graph.
#
# OFX has no standard way for a plug-in to choose its node colour, so Nuke has to
# be told from Python. Installation:
#   1. Copy this file into  %USERPROFILE%\.nuke\
#   2. Add this line to     %USERPROFILE%\.nuke\menu.py   (create the file if needed):
#          import push_it_nuke
#
# The colour is applied when a push_IT_OFX node is created. A colour changed by
# hand (tile_color != 0) is kept, including when a script is reopened.

import nuke

# Node colour (0..1), edit here if needed.
PUSHIT_COLOR = (0.7, 0.592, 0.21)

# Nuke names OFX node classes "OFX" + plug-in identifier + "_v" + major version.
_CLASS_PREFIX = "ofxfr.pushit.push_it_ofx"


def _to_tile_color(rgb):
    r, g, b = [max(0, min(255, int(round(c * 255.0)))) for c in rgb]
    return (r << 24) | (g << 16) | (b << 8) | 0xFF


def _set_push_it_color():
    node = nuke.thisNode()
    if not node.Class().lower().startswith(_CLASS_PREFIX):
        return
    knob = node.knob("tile_color")
    if knob is not None and knob.value() == 0:      # 0 = default colour, never changed by the user
        knob.setValue(_to_tile_color(PUSHIT_COLOR))


nuke.addOnCreate(_set_push_it_color)
