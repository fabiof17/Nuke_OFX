# Nuke OFX Plug-ins

A collection of OpenFX plug-ins for The Foundry Nuke, written in C++ with optional NVIDIA CUDA acceleration.

| Plug-in | What it does |
|---|---|
| [**push_IT_OFX**](#push_it_ofx) | Edge extend / pixel spread: pushes the colours of a matte outward, or fills a matted area from its surroundings. |
| [**apGlow_OFX**](#apglow_ofx) | Exponential glow: a sum of blurs whose sizes grow geometrically, for a soft and wide light falloff. |
| [**L_Despill_OFX**](#l_despill_ofx) | Spill suppression for green, blue and red screens, with spill matte processing, spill replace and background re-light. |
| [**despill_IT_OFX**](#despill_it_ofx) | Spatially aware despill: local screen colour, despill fading from the edges inwards, protected hues, edge colour rebuilt from the subject, background light and light wrap. |
| [**CornerPinEx_OFX**](#cornerpinex_ofx) | GPU-accelerated corner pin derived from Natron's CornerPin: CUDA rendering, EWA high-quality filters and per-corner offsets. GPL v2+. |
| [**ChromaAberrations_OFX**](#chromaaberrations_ofx) | Physically based lens aberrations: wavelength-by-wavelength PSFs of a parametric or real (ray-traced) lens, with variable blur, diffraction, chromatic aberration, distortion and vignetting. GPL v3 or commercial. |

The plug-ins run on Windows, Linux and macOS. On Windows and Linux they can render on an NVIDIA GPU with CUDA; when no GPU is available they fall back to a multithreaded CPU renderer automatically. The CPU and GPU renders are identical (CornerPinEx_OFX: within 1 to 2 levels out of 255 with the classic filters; ChromaAberrations_OFX: see its section).

Ready-to-use builds for Windows and Linux are available on the [Releases](../../releases) page.

Each plug-in lives in its own folder and is built on its own:

```
push_IT_OFX/      edge extend plug-in (sources, CMakeLists.txt, build.bat, build.sh)
apGlow_OFX/       glow plug-in (sources, CMakeLists.txt, build.bat, build.sh)
L_Despill_OFX/    classic despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
despill_IT_OFX/   spatially aware despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
CornerPinEx_OFX/  GPU corner pin plug-in (sources, tests, CMakeLists.txt, build.bat, build.sh, install checks)
ChromaAberrations_OFX/  optical aberrations plug-in (sources, lenses, presets, tools, tests, CMakeLists.txt, build.bat, build.sh)
ofx_bench/        optional stand-alone CPU / GPU benchmark (no Nuke needed)
```

---

## push_IT_OFX

Edge extend for compositing, similar to the *Pixel Spread* tool of Autodesk Flame.

- **Expand** extends the colours of the matte outward: edge extend before a keyer, a despill, or a merge over a new background.
- **Contract** fills the matted area with the pixels around it: quick clean plates, object removal.

### Inputs

| Input | Role |
|---|---|
| **Source** | Image to process. |
| **Matte** | Optional. Defines the area to keep. Without it, the alpha of Source is used. |
| **Vector** | Optional. Direction vectors for Vector Warp (R = X, G = Y, in pixels). |

### Parameters

**GPU** (first line of the panel)
- **Use GPU**: renders on the NVIDIA GPU. The detected card is shown on the right ("Local GPU: ...").

**Spread**
- **Spread Type**
  - **Interpolate**: smooth fill using a push-pull pyramid (a clean edge extend).
  - **Stretch**: every pixel takes the colour of the nearest edge pixel (exact distance transform). Keeps the grain and contrast of the edge.
  - **Vector Warp**: directional stretch along the Vector input or a constant angle. Pixels that find nothing in that direction are completed with Interpolate.
- **Spread Mode**: Expand or Contract.
- **Max Distance** / **Distance Falloff**: limit the spread to N pixels, with a soft end (0 = unlimited).
- **Spread Blur**: blur applied to the spread pixels only.
- **Seed Threshold**: matte value from which a pixel can be copied (Stretch, Max Distance).

**Matte**
- **Matte Source**: Auto, Source Alpha, Matte Alpha, Matte Luminance.
- **Invert Matte**.
- **Source Premultiplied** (off by default): turn it on when the source is premultiplied; in Expand, the source is then unpremultiplied so that the edges do not darken.
- **Edge Shrink**: erodes the kept area by N pixels to skip contaminated edge pixels (very useful in Contract).
- **Edge Softness**: softens the transition between original and spread pixels.

**Vector Warp**
- **Vector Source**: Vector Input or Constant.
- **Angle** / **Length** (Constant): direction (0° = right, 90° = up) and length of the stretch.
- **Motion Distance**: gain on the input vectors (negative = opposite direction).
- **Motion Threshold**: ignores vectors shorter than this.
- **Vector Origin** and **Red / Green / Blue X / Y Gain**: matrix that turns the RGB channels of the Vector input into an X / Y displacement.
- **Invert Direction**.

**Output**
- **Result Output**: Combined (original + spread, unpremultiplied RGB), Spread Only, or Premult (Combined premultiplied by the chosen output alpha).
- **Alpha Output**: Input, Spread Matte (extended matte), Solid.
- **Extend BBox to Format**: grows the bounding box to the format so that the spread reaches the image borders.

**Viewer**
- **Show Widgets**: while the node panel is open, the viewer shows:
  - with *Vector Warp* and a constant vector, an **arrow** for *Angle* and *Length*: drag its tip to set both;
  - with *Max Distance* > 0, a **dashed circle** of that radius: drag its handle to change it;
  - a **centre cross** that moves these widgets (they only show sizes and directions, their position does not change the render).

### Recipes

- **Edge extend before a despill or a merge**: premultiplied plate in Source, Expand, Interpolate, Edge Shrink 1–2, Alpha Output = Input, then premultiply.
- **Quick clean plate**: plate in Source, object roto in Matte, Contract, Interpolate or Stretch, Edge Shrink 2–5, Spread Blur to taste.
- **Directional stretch**: Vector Warp with Vector Source = Constant, angle set to the direction the pixels should flow. For local control, connect motion vectors (in pixels) to the Vector input.

---

## apGlow_OFX

Exponential glow built with the multi-blur method: several blurs whose sizes grow geometrically up to *Max Size* are added together, with a colour gain that changes progressively from the smallest blur to the largest. The result imitates the soft, wide falloff of real light diffusion.

This is a C++ / CUDA re-implementation of the **apGlow** gizmo by Adrian Pueyo (apGlow v1.2, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022), calibrated to match it.

Formerly named expo_GLOW_OFX. The plug-in identifier has not changed, so Nuke scripts made with expo_GLOW_OFX still load; their existing nodes keep the name they were given.

### Inputs

| Input | Role |
|---|---|
| **Source** | Image to glow. |
| **Area** | Optional. Area that emits the glow: alpha, or luminance for an RGB image. |
| **Mask** | Optional. Limits the final effect. |

### Parameters

**GPU** (first line of the panel)
- **Use GPU**: renders on the NVIDIA GPU. The detected card is shown on the right.

**Glow**
- **Intensity**: glow multiplier.
- **Persistence** (RGB): colour gain of the largest blur. It acts progressively on the others and hardly on the smallest one, as an inverse falloff or a progressive tint.
- **Tint** (RGB): global colour multiplier.
- **Effect Only**: outputs the glow alone. Off: the glow is added to the image. The source alpha is always kept.

**Source response** (the pixel colour is kept, only its luminance goes through this curve)
- **Src Black** / **Src White**: black and white points of the luminance. Nothing glows below the black point.
- **Response**: gamma of the luminance (shape of the curve).
- **Clamp**: limits the luminance to 1.

**Size**
- **Size Scale**: multiplies the size of every blur.
- **Max Size**: size of the largest blur, in pixels.
- **Size Falloff**: lower = more diffuse glow, higher = tighter glow.
- **Steps**: number of blurs. Blur *i* (1 to Steps) has the size `(MaxSize × Falloff)^(i / Steps) / Falloff`.
- **Filter**: box, triangle, quadratic or gaussian, with the same sizes as Nuke's Blur node.
- **Quality**: like Nuke's Blur (15 by default). Blurs larger than this are computed on a reduced image and scaled back up, which is much faster and softens the filter like Nuke does. 0 = exact filter.

**Bounding box**
- **Crop to Format**: crops the result to the format.
- **Overscan** / **Enable Overscan**: keeps the glow up to N pixels around the format.
- When the result is cropped, a 1-pixel black border is added, like a Nuke Crop with *black outside*.

**Area / Mask**
- **Invert Area**, **Invert Mask**, **Mix**.

**Viewer**
- **Show Widgets**: while the node panel is open, the viewer shows one circle per glow step (diameter = blur size of the step, *Size Scale* included), the largest one solid. Drag its handle to set *Max Size*. The widget stays at the centre of the format.

As in Nuke, the blurs see the source (and the Area input) with their edge pixels repeated beyond the bounding box, so the glow does not darken near the image borders.

---

## L_Despill_OFX

Spill suppression for keying. The screen channel is limited by a mix of the two other channels; the amount of spill removed gives a **spill matte**, which can be filtered and then drives a grade of the despilled area, a grade of the foreground and the re-light of the edges with a background.

It brings together the features of two widely used gizmos, **L_Despill v05** by Justin Johnson (Luma Pictures) and **DespillMadness**, with their inconsistencies fixed (see below).

### Inputs

| Input | Role |
|---|---|
| **Source** | Plate to despill. |
| **BG** | Optional. Background used to re-light the despilled edges. |
| **Mask** | Optional. Limits the effect. |

### Parameters

**GPU** (first line of the panel)
- **Use GPU**: renders on the NVIDIA GPU. The detected card is shown on the right.

**Despill**
- **Screen Color**: Green, Blue or Red.
- **Despill Math**
  - **Average**: the screen channel is limited by a weighted average of the two other channels.
  - **Max**: limited by the larger of the two other channels (gentler on colours close to the screen).
- **Bias**: weight between the two other channels. For a green screen, −1 = red limits green, +1 = blue limits green, 0 = equal weights, −0.33 = double red average (2R + B) / 3.
- **Limit Scale**: multiplies the limit. Below 1: stronger despill; above 1: lighter.
- **Math Colorspace**: Log (Cineon) or Linear. The image stays linear in and out.
- **Balance White** / **Balance Black**: white and black point of the screen channel before the despill, reverted afterwards. Use them when the whole plate has a cast of the screen colour.
- **Restore Luminance**: adds back the luminance removed with the spill, so that despilled areas do not get darker.

**Spill matte**
- **Matte Colorspace**: Linear (default), Log, or the same as the despill.
- **Matte Blackpoint** / **Whitepoint** / **Gamma**: grade of the spill matte.
- **Median Size**: median filter radius, removes noise and grain (up to 10 px).
- **Clean Size**: positive values fill small holes (dilate then erode); negative values remove small specks (erode then dilate).
- **Matte Blur**: blur of the spill matte.

**Spill replace**: grade (blackpoint, whitepoint, lift, gain, multiply, offset, gamma) and saturation of the despilled area, inside the spill matte, in Log or Linear.

**FG grade**: the same grade and saturation, applied to the foreground outside the spill matte.

**Background** (BG input)
- **BG Mode**
  - **Add**: the graded BG is multiplied by the spill matte and added, re-lighting the edges with the background.
  - **Keymix**: the BG replaces the despilled image inside the spill matte.
  - **Off**: the BG input is ignored.
- **BG Colorspace**, **BG Blackpoint** / **Whitepoint** / **Gamma** / **Saturation**, **BG Black Clamp** / **White Clamp**: grade of the BG for the Add mode.

**Output**
- **Alpha Output**: source alpha, or the spill matte.
- **Invert Mask**, **Mix**.

The spill matte filters see the source with its edge pixels repeated beyond the bounding box, as Nuke's filters do, so the matte does not fade at the image borders.

### Differences from the original gizmos

- **Despill maths unified**: the 10 formulas of the two gizmos are covered by *Average* / *Max* with *Bias* and *Limit Scale*. In L_Despill, *Fine Tune* changed meaning from one mode to the next (channel weighting or limit scale); here each setting does one thing.
- **Same formula for the colour and the matte**: in DespillMadness the double-average modes removed one amount of spill from the colour but wrote another one into the matte, and the blue-screen version divided by 2 instead of 3.
- **No limit on the spill removed**: L_Despill clamped the spill to 1 in its Average and Max modes, which left very strong spill partly untouched.
- **Exact channels**: only the screen channel is modified. L_Despill converted all channels to log and back, adding rounding errors to the other two.
- **Restore luminance** (from DespillMadness) no longer clips at 1, so it works on HDR plates.
- **Log conversion protected** against negative values, which produced invalid pixels (NaN).
- **BG Add does nothing without a BG input**. In L_Despill, the default BG grade applied to an empty input darkened the despilled areas.
- **Spill matte clean**: L_Despill's *AlphaClean* group is replaced by a standard morphological clean (fill holes / remove specks).
- **Spill matte in Linear by default**: a matte measured in log space is very weak, so the spill replace grade barely acted.

---

## despill_IT_OFX

A spatially aware despill. A classic despill (such as L_Despill_OFX) treats every pixel on its own, with one screen colour for the whole image. As a result, it also removes wanted colours (green eyes, teal clothes, olive skin) and turns semi-transparent edges grey. despill_IT_OFX uses the image around each pixel:

- **Local screen**: the spill is measured against the screen colour *around* the pixel, estimated from the screen areas of the plate and interpolated behind the subject, or read from a clean plate. Uneven screens are handled.
- **Spatial strength**: the despill is full near the screen and fades towards the inside of the subject, using the exact distance of every pixel to the screen.
- **Protection**: a hue range (skin, eyes...) and/or a Protect matte are left untouched.
- **Edge colour**: the colour of semi-transparent edges (hair, motion blur) is rebuilt from the opaque core of the subject, propagated outwards, keeping the luminance of the edge. Hair stays blond instead of turning grey.
- **Spill light and light wrap**: the screen light removed with the spill can be replaced by the light of the blurred background, and the blurred background can be wrapped on the inner edges of the subject.

### Inputs

| Input | Role |
|---|---|
| **Source** | Plate to despill. |
| **Key** | Optional. Alpha of a keyer (1 = subject). Without it, an internal key is computed from the amount of screen in each pixel. |
| **CleanPlate** | Optional. Plate of the empty screen, used as the local screen colour. |
| **BG** | Optional. Background for the spill light and the light wrap. |
| **Protect** | Optional. Matte of areas to leave untouched. |
| **Mask** | Optional. Limits the effect. |

### Parameters

**GPU** (first line of the panel)
- **Use GPU**: renders on the NVIDIA GPU. The detected card is shown on the right.

**Screen**
- **Screen Color**: colour of the screen, picked on an evenly lit part of it. Its largest component gives the screen channel (green, blue or red).
- **Screen Source**: Global Color (the Screen Color everywhere), From Plate (local estimate, default) or Clean Plate.
- **Despill Math** (Average / Max), **Bias**, **Limit Scale**: how the spill is detected, as in L_Despill_OFX.

**Key**
- **Key Source**: Auto (the Key input when connected, otherwise the internal key) or Internal.
- **Key Low** / **Key High**: internal key, fraction of screen at which a pixel becomes partly, then fully, screen.
- **Screen Below** / **Core Above**: key values that define the screen (used for the local screen and the distances) and the opaque core (whose colour is propagated into the edges).

**Strength**
- **Strength**: overall despill amount.
- **Spatial**, **Edge Width**, **Edge Falloff**, **Interior Strength**: full despill up to *Edge Width* pixels from the screen, then a fade over *Edge Falloff* pixels down to *Interior Strength*, which is the amount kept for spill reflected deep inside the subject.

**Protection**
- **Protect Hue**, **Protect Range**, **Protect Softness**: hue range to protect, in degrees (30 = skin; range 0 = off).
- **Protect Amount**: strength of the protection, from the protected hues and the Protect input.

**Despill**
- **Despill Mode**
  - **Screen Channel** (default): the screen channel is limited by the two others, the classic despill but relative to the local screen.
  - **Neutralize**: the screen colour becomes neutral light of the same luminance.
  - **Subtract Screen**: the screen light is removed. This is physically right for reflected spill. Edges get darker, as in a premultiplied foreground, so combine it with *Spill Light = Background*.
- **Edge Restore**: amount of the edge colour rebuilt from the opaque core.

**Background** (BG input)
- **Spill Light**: None, or Background (the removed screen light is replaced by the blurred BG light).
- **BG Gain**, **BG Blur**: level and blur of the BG light.
- **Light Wrap**: adds the blurred BG on the inner edges of the subject.

**Output**
- **Output**: the result, or a view to set up the effect (spill map, despill strength, key, local screen, propagated core colour).
- **Alpha Output** (source alpha, key, spill map), **Invert Mask**, **Mix**.

**Viewer**
- **Sample Screen** (under Screen Color) and the **sample area** drawn in the viewer: move its corners or its centre, or Shift + drag anywhere to draw a new area. On release, *Screen Color* becomes the average colour of the Source inside the area; the button samples it again (after a frame change, for instance). A small swatch next to the area shows the current Screen Color.
- **Show Widgets**: shows the sample area and, with *Spatial* on, a **distance probe**: a solid circle of radius *Edge Width* and a dashed one of radius *Edge Width + Edge Falloff*. It stays at the centre of the format; drag its handles to change the two values.
- **Show Overlay** (needs *Spatial*): tints in semi-transparent blue the pixels of the subject that get the edge despill, from the screen up to *Edge Width + Edge Falloff*: 50 % over *Edge Width*, fading to 20 % at the end of *Edge Falloff*. Unlike the widgets, the tint is part of the rendered image: turn it off before rendering.

### How to use

The defaults come from tests on a synthetic plate. Treat them as starting points to adjust on real plates.

**1. Connect the node**

```
Plate ──────────────┬──► despill_IT_OFX (Source) ──► … ──► Premult ──► Merge (over) ──► BG
Keyer (alpha) ──────┘    (Key, optional)
BG ───────────────────►  (BG, optional)
```

- **Source**: the raw plate, not premultiplied.
- **Key** (recommended): the output of a keyer (IBK, Keylight, Primatte…). Only its alpha is read (1 = subject). Without it, an internal key is used, which is enough to start.
- **BG** (optional): the final background, for the spill light and the light wrap.
- **CleanPlate** (optional): a plate of the empty screen, if you have one.
- **Protect** (optional): a roto of areas to leave untouched (green eyes, for example).

The node outputs a despilled, unpremultiplied image. Keying and premultiplication happen afterwards, as usual.

**2. Pick the screen colour**

Place the **sample area** shown in the viewer on a clean, evenly lit part of the screen (Shift + drag to draw it): Screen Color is set to its average when you release the mouse. The **Screen Color** colour picker works too. This is the most important setting: it gives the reference hue and the channel to treat (green or blue).

Leave **Screen Source** on *From Plate*: the screen colour is then estimated around every pixel, which handles uneven screens. To check it, set **Output** to *Local Screen*: you should see the screen rebuilt behind the subject, with no trace of the subject.

**3. Check the key**

Set **Output** to *Key*: the subject should be white, the screen black.
- With a Key input, you see the alpha of your keyer.
- Without one, adjust **Key Low** and **Key High** until the screen is black and the subject white. This key does not need to be perfect: it only locates the screen and the core of the subject, it is not used for the final composite.

**4. Set where the despill acts (the key step)**

Set **Output** to *Strength*: white = full despill, black = none.
- **Edge Width** (4 px): distance from the screen over which the despill is full.
- **Edge Falloff** (40 px): distance over which it fades towards the inside of the subject.
- **Interior Strength** (0.15): what remains in the core of the subject. Raise it if reflected spill remains far from the edges (green on clothes, under the chin…); lower it if wanted colours get damaged.

The *Spill Map* view shows the amount of spill actually removed.

**5. Protect colours**

If a colour close to the screen must stay untouched, use the *Protection* settings:
- **Protect Hue**: hue to protect, in degrees (30 = skin, 180 = teal).
- **Protect Range**: its width (0 = off). Start around 10–20.
- **Protect Softness**: softness of the transition.

For a specific area, connect a roto to the **Protect** input instead.

**6. Choose the despill mode**

- **Screen Channel** (default): classic despill, but relative to the local screen. The safest choice.
- **Neutralize**: the screen colour becomes neutral grey of the same brightness. Edges stay bright.
- **Subtract Screen**: removes all of the screen light. Physically right for reflected spill, but edges get darker; combine it with *Spill Light = Background*.

**Edge Restore** (0.75) rebuilds the colour of semi-transparent edges (hair, motion blur) from the core of the subject, so that hair does not turn grey. The *Core Colour* view shows the propagated colour.

**7. Blend with the background (when BG is connected)**

- **Spill Light = Background**: the removed screen light is replaced by the light of the blurred background, so the subject takes the colour of its new surroundings.
- **BG Gain**: intensity of that light. **BG Blur**: softness of the background used.
- **Light Wrap** (0 by default): adds the blurred background on the inner edges of the subject. Keep it subtle: 0.1–0.3 is usually enough.

**8. Finishing**

- **Strength**: overall despill amount.
- **Alpha Output**: outputs the key or the spill map in the alpha, to use them further down the script.
- **Mix** and **Mask**: as on any Nuke node.

**Quick recipes**

| Situation | Settings |
|---|---|
| Simple shot, clean screen | Screen Color with the picker, everything else at default |
| Uneven screen (vignetting, folds) | *Screen Source* on From Plate (default), or Clean Plate if you have one |
| Blond hair turning grey | Edge Restore → 1, Despill Math → Max |
| Green or teal clothes | Interior Strength → 0, or Protect Hue on their hue |
| Reflected spill in the middle of the subject | Interior Strength → 0.4–0.6 |
| Strong integration with the background | BG connected, Spill Light = Background, Light Wrap 0.2 |

### Compared with a classic despill

Measured on a synthetic plate: an uneven green screen, a subject with skin, blond hair with soft edges, green spill reflected near the edges, and a teal shirt. The figures are the mean error against the true colours (lower is better), with default settings.

| Area | No despill | Classic despill (L_Despill_OFX, linear average) | despill_IT_OFX |
|---|---|---|---|
| Teal shirt | 0.000 | 0.067 | 0.015 (0.000 with *Protect Hue*) |
| Blond hair | 0.001 | 0.024 | 0.004 |
| Semi-transparent edges (colour) | 0.297 | 0.138 | 0.087 |
| Spill reflected near the edges | 0.021 | 0.024 | 0.022 |

Like every despill that compares colour channels, it detects reflected spill poorly on colours that naturally lack the screen colour, such as green on skin. For that case, raise *Interior Strength* or use *Subtract Screen*.

---

## CornerPinEx_OFX

A GPU-accelerated corner pin, derived from **Natron's CornerPin** ([openfx-misc](https://github.com/NatronGitHub/openfx-misc), `CornerPin/CornerPin.cpp`, and [openfx-supportext](https://github.com/NatronGitHub/openfx-supportext), `ofxsTransform3x3`). It keeps everything the original does and adds:

| Addition | Details |
|---|---|
| **GPU rendering (CUDA)** | Filtering, motion blur and mask / mix run on the NVIDIA GPU. If the GPU is not available or an error occurs, the frame is rendered on the CPU automatically. |
| **High-quality filters** | **EWA Gaussian** and **EWA Lanczos3**: anisotropic filters computed on a mipmap pyramid. They avoid aliasing and moiré even under strong perspective. They run on the GPU, and on the CPU as a fallback. |
| **Per-corner offsets** | Every "to" and "from" point has its own x and y offset, in a collapsible *Offsets* group of the *to* and *from* tabs. Useful to adjust tracked corners without touching the tracking curves. The overlay draws the offset quad in orange. |
| **GPU settings** | Below *Overlay Points*, between two separators: the `Use GPU (CUDA)` checkbox, with the detected GPU model on its right. The first NVIDIA GPU of the machine is used. |

Both variants of the original are provided: **CornerPinEx_OFX** and **CornerPinExMasked_OFX** (with a mask input and a mix). Their identifiers differ from Natron's (`community.openfx.CornerPinEx_OFX…`), so both plug-ins can be installed side by side.

> **License**: unlike the rest of this repository (MIT), CornerPinEx_OFX is derived from GPL code and is therefore released under the **GPL v2 or later** (see [`CornerPinEx_OFX/LICENSE`](CornerPinEx_OFX/LICENSE)). You can use it freely, in production too. If you distribute the plug-in (binaries included), you must also provide its source code under the same license. The modified files say so in their header.

### Parameters

The parameters added to Natron's CornerPin:

- **Filter**: on top of Natron's filters, two HQ filters:
  - `EWA Gaussian (HQ)`: the cleanest under strong perspective, no ringing, slightly soft;
  - `EWA Lanczos3 (HQ)`: sharp and alias-free. **Turn on `Clamp`** to remove ringing on high-contrast edges.
- **Offsets** (collapsible group in the `to` and `from` tabs): `to1 offset x`, `to1 offset y` … `from4 offset y`, each animatable on its own, and a `Reset Offsets` button per tab. The offsets are part of the point positions, so `Amount` interpolates them too.
- **Overlay Points** switches by itself: changing a value of the `To` tab (point, enable checkbox, offset, `Copy "From"` and `Reset Offsets` buttons) shows the "to" points, and a value of the `From` tab shows the "from" points. Just clicking a tab is not enough, because Nuke does not tell OFX plug-ins about tab changes. The menu can still be set by hand.
- **GPU** (below *Overlay Points*): `Use GPU (CUDA)` and the GPU model on its right. If a GPU render fails (out of memory, for instance), the frame is rendered again on the CPU, and the message is written to the console and shown next to the GPU name.
- **Crop to Format** (on by default): limits the output to the project format. When the corner pin enlarges the image past the frame, the pixels outside the format are neither computed nor transferred. Turn it off to keep the overscan.

All the other parameters (from / to, Amount, Extra Matrix, Invert, Motion Blur, Shutter, Black outside, Mask / Mix…) behave as in Natron's CornerPin.

### How it works

Nuke gives images to OFX plug-ins in CPU memory: it does not offer the OpenFX 1.5 GPU render paths (CUDA / Metal / OpenCL) that DaVinci Resolve does, for instance. So the plug-in manages the GPU itself, for every render:

1. it uploads the source (and the mask) to the GPU and converts it to float;
2. for the EWA filters, it builds the mipmap pyramid on the GPU;
3. a single kernel computes every pixel of the render window: homography, filter, adaptive motion blur and mask / mix;
4. it converts the result to the output bit depth and copies it back into Nuke's image.

The plug-in asks Nuke for whole images (no tiles), so there is one upload per frame. The GPU resources (stream, device memory, page-locked transfer buffers) are kept from one render to the next in two "slots": at most two renders run on the GPU at the same time. Transfers alternate between two page-locked buffers, so that the CPU-side copy and the PCIe transfer overlap.

**The sampling code is the same on the CPU and on the GPU** (`src/cpSampler.h`, compiled both as C++ and as CUDA). The classic filters (Impulse … Notch) use exactly Natron's formulas, and the motion blur uses the same algorithm: adaptive Monte Carlo over 1000 transforms of the shutter interval.

When the GPU is off and the filter is a classic one, **Natron's original CPU code** runs, and the result is identical, pixel for pixel, to Natron's CornerPin (see *Tests*). On the GPU, the classic filters follow the same algorithm, including Natron's 3ⁿ×3ⁿ supersampling when minifying, but compute in float instead of double: they differ from Natron by at most 1 to 2 levels out of 255, on fewer than 0.3 % of the pixels.

### When does the GPU help?

Because of the CPU ↔ GPU transfers that Nuke imposes, the gain is small for a plain Cubic without motion blur. It is large with **motion blur** (up to 40 × `Motion Blur` samples per pixel) and with the **EWA filters** (dozens of texels per sample). On one CPU thread, a 2048×1080 RGBA float frame takes about 0.3 s in Cubic, 3.7 s in EWA Lanczos3, 15 s in Cubic with motion blur and 130 s in EWA with motion blur: exactly the cases the GPU parallelises. The CPU renderer of course uses every core. `test_engine --bench` prints the CPU and GPU times on your machine.

### Measuring performance

Set the environment variable `CORNERPINEX_OFX_PROFILE=1` before starting Nuke:

```bat
set CORNERPINEX_OFX_PROFILE=1
"C:\Program Files\Nuke16.0v6\Nuke16.0.exe"
```

On Linux: `CORNERPINEX_OFX_PROFILE=1 nuke`.

Each render appends a line to `%TEMP%\CornerPinEx_OFX_profile.log` (Windows) or `/tmp/CornerPinEx_OFX_profile.log` (Linux), with the rendered size, the filter and the time of each stage: upload, mipmaps, compute, copy back to Nuke. CPU renders are logged too. For transfers, `cpu copy` is the time spent copying images on the CPU side. These copies are spread over several cores (half of the cores, 6 at most) by threads created once. `CORNERPINEX_OFX_COPY_THREADS=n` forces the number of threads, and `1` turns the spreading off.

`CORNERPINEX_OFX_TRANSFER` picks how images move between Nuke's memory and the GPU:

| Value | Method |
|---|---|
| `staged` (default) | CPU copy into two page-locked buffers, alternating with the transfer |
| `direct` | direct transfer from Nuke's memory; the NVIDIA driver manages the buffers itself |
| `register` | Nuke's image memory is page-locked for the duration of the transfer: no CPU copy (`page-lock` in the log) |

The fastest one depends on the system and the machine: compare them with the log. On the Windows test machine (2K, Cubic, motion blur), `staged` was the fastest overall: about 40 ms per frame, against 45 to 65 ms with `direct` and 70 to 95 ms with `register`. With `register`, the transfer itself was a little shorter, but the GPU compute became 4 to 5 times slower. On Linux, `register` may well be the fastest: measure it.

### Tests

- `test_engine`, the engine test executable, needs no OFX host. It checks the engine on known cases: identity, translation, anti-aliasing of a checkerboard minified ×8 in perspective, motion blur ramp, mask / mix, 8- and 16-bit depths, tiled rendering. **If a CUDA device is present, it also compares the GPU render with the CPU render** for every filter, bit depth (8 / 16 / 32 bits) and channel count (1 to 4), with and without motion blur. `test_engine --bench` also prints the render times. `build.bat` and `build.sh` build and run it.
- `tests/natron_render_test.py` and `tests/compare_renders.py` test the plug-in in a real OFX host (Natron on the command line):

  ```sh
  CPGPU_OUT=/tmp/out OFX_PLUGIN_PATH=build NatronRenderer -t tests/natron_render_test.py
  python3 tests/compare_renders.py /tmp/out
  ```

  With the classic filters, CornerPinEx_OFX gives exactly the same image as Natron's CornerPin (0 pixels differ), with the offsets too (compared with Natron's CornerPin given the already-summed points) and with motion blur.

### Building and checking the installation

CornerPinEx_OFX builds like the other plug-ins (see [Building](#building)), with a few differences:

- CMake downloads the OFX headers and support library from [NatronGitHub/openfx](https://github.com/NatronGitHub/openfx) and [openfx-supportext](https://github.com/NatronGitHub/openfx-supportext), pinned to the commits used by openfx-misc.
- `build.bat` and `build.sh` also build and run `test_engine`. When `build.bat` is run as administrator, it offers to install the bundle into the OFX folder. `build.sh install` copies it into `/usr/OFX/Plugins` (sudo).
- `check_install.bat` (Windows) and `./check_install.sh` (Linux) check the installation, load the plug-in as Nuke does, report missing libraries and clear Nuke's OFX cache. Close Nuke before running them.
- CMake options: `CORNERPINEX_OFX_WITH_CUDA` (`ON`), `CORNERPINEX_OFX_BUILD_TESTS` (`OFF`), `CMAKE_CUDA_ARCHITECTURES` (default `"75-real;86-real;89-real;75-virtual"`; the build scripts add the installed GPU if it is newer, e.g. `120-real` for an RTX 50xx with CUDA 12.8+), and `OFX_DIR` / `SUPPORTEXT_DIR` to use local checkouts instead of downloading them.
- On Linux, **build on the machine that runs Nuke**, or on a distribution at least as old: the plug-in depends on the glibc version of the build system. If nvcc rejects your GCC version ("unsupported GNU version"), point it to a compatible one: `CUDAHOSTCXX=g++-12 ./build.sh clean` (or enable a suitable `gcc-toolset` on Rocky).
- The GitHub Actions workflow `.github/workflows/cornerpinex_windows.yml` builds the Windows plug-in (MSVC + CUDA), runs the CPU tests and uploads the bundle as an artifact.

The node is in the **Transform** menu. Next to `Use GPU (CUDA)`, the panel should show your card, e.g. `Local GPU: NVIDIA GeForce RTX 4080 (16376 MB)`. "CUDA: no NVIDIA GPU available" means rendering will run on the CPU.

### Files

```
src/CornerPinEx_OFX.cpp   plug-in (Natron's CornerPin.cpp + GPU, offsets, HQ filters; changes marked "CornerPinEx_OFX")
src/cpTransform3x3.*      modified copy of ofxsTransform3x3 (prepareRender hook + HQ filter options)
src/cpSampler.h           sampling shared by CPU and GPU: filters, EWA, motion blur, mask / mix
src/cpCuda.cu             CUDA engine
src/cpCpuEngine.*         CPU engine (HQ filters without GPU, test reference)
src/cpCudaStub.cpp        replaces cpCuda.cu in a build without CUDA
tests/                    engine tests and render tests in Natron
resources/                node icons
```

---

## ChromaAberrations_OFX

Physically based optical aberrations. The plug-in computes the point spread function (PSF) of a lens wavelength by wavelength, on a grid covering the image, and applies it to the plate:

- blur that varies across the field;
- diffraction;
- axial and lateral chromatic aberration;
- distortion;
- vignetting.

The lens is described in one of two ways:

- **Parametric**: you set its wavefront by hand (Seidel aberrations and Zernike terms), its diaphragm and its sensor;
- **Real Lens**: you pick a real optical prescription (glasses, radii of curvature, thicknesses), traced ray by ray and wavelength by wavelength.

Rendering runs on the GPU (CUDA + cuFFT) when possible, otherwise on the CPU. Two command-line tools, **`carender`** (applies the lens to an EXR) and **`psfmap`** (draws the PSFs), are used for tests and tuning outside Nuke.

> **License**: unlike the rest of this repository (MIT), ChromaAberrations_OFX is **dual-licensed**: GPL v3 ([`ChromaAberrations_OFX/LICENSE`](ChromaAberrations_OFX/LICENSE)) or a commercial license for studios and vendors who want to integrate or distribute it without the GPL obligations. See [`ChromaAberrations_OFX/LICENSING.md`](ChromaAberrations_OFX/LICENSING.md) for the details and the third-party components and data. To keep the dual license possible, outside contributions need a contributor agreement (CLA).

### Inputs

| Input | Role |
|---|---|
| **Source** | Plate to process, in linear light. |
| **Mask** | Optional. Limits the effect (with `Mix` and `Invert Mask`). |

### Parameters

The settings are split over five tabs.

**Lens**
- `Preset`: a ready-to-use lens (see [Cameras and presets](#cameras-and-presets)).
- `f-number`: working aperture f/N.
- `Filmback Width (mm)`: width of the sensor covered by the image (Super 35: 24.89; full frame: 36). It sets the size of a pixel, and therefore the size of the PSFs.
- `Lens Model`, `Lens`, `Lens File` and `Focus Distance`: choice of a real lens (see [Real lenses](#real-lenses)).
- `Focal Length (mm)`: focal length of the real lens, which is rescaled to it. In parametric mode, it drives the natural cos⁴ falloff.
- `Centre on Image` (on by default): the optical axis is at the centre of the image. Turn it off to place the axis with the `Optical Center` handle in the Viewer, for a cropped plate for instance.
- Diaphragm: `Diaphragm Blades`, `Blade Roundness`, `Blade Rotation` and `Pupil Aspect` (oval bokeh).
- `Mechanical Vignetting` and `Vignetting Start`: cat's-eye pupil in the corners.
- `Natural Vignetting`: amount of the cos⁴ falloff.

**Aberrations**
- The coefficients are in waves, at the edge of the pupil and in the corner of the image.
- Available fields: Spherical, Coma, Astigmatism, Field Curvature, Distortion, Defocus, Axial Colour, Lateral Colour, Spherochromatism.
- `Zernike Terms`: additional terms, e.g. `9:0.5, 11:0.3`.
- `Aberration Amount` multiplies every coefficient. At 0, only diffraction remains.
- The defaults give a decent modern lens, barely imperfect.

**Spectral**
- `Colour Space`: ACEScg (default), linear Rec.709 or linear Rec.2020. Images must be in linear light.
- `Camera`: spectral sensitivities of the camera that shot the plate (default: CIE 1931 observer), and `Camera File` for a custom file.
- Number of wavelengths in preview (8) and in final (24).

**Render**
- `Quality`:
  - `Preview`: `PSF Grid (Preview)` grid (7 × 5), 8 wavelengths;
  - `Final`: 24 wavelengths, automatic or fixed grid (below);
  - `Preview in Viewer, Final in Render`: Preview in the Viewer, Final when rendering to disk. It relies on the "interactive render" flag the host passes to the plug-in; a host that does not pass it always gets Final.
- `Final Grid`:
  - `Automatic` (default): the grid starts from the Preview grid and is refined until the PSF interpolated at the centre of a cell differs from the PSF computed there by less than `Grid Tolerance` (2 % by default). The grid is limited to 25 points per side;
  - `Fixed`: `PSF Grid (Final)` grid (15 × 11).
- `Max PSF Size` (129 by default) and `Auto Size` (on by default). The size of each PSF is derived from the lens: the largest geometric spot, plus a margin for diffraction. `Max PSF Size` is then only a ceiling, in pixels at full resolution, so a sharp lens stays fast: 9 to 19 pixels instead of 129. Without `Auto Size`, every PSF is `Max PSF Size`.
- `Use GPU (CUDA)`, with the GPU model shown on the right.

**Output**
- `View`:
  - `Image`: the plate through the lens;
  - `PSF Map`: the PSFs in white light at their place in the image, enlarged (`PSF Zoom`, 0 = automatic). It is displayed with a 2.2 gamma: it is a tuning view, not linear light. `Image Behind` shows the darkened image behind the PSFs; turned off, the background is black;
  - `Test Chart`: the lens applied to a built-in chart (Siemens stars and HDR light points).
- `Blur`, `Distortion & Lateral Colour` and `Vignetting` checkboxes turn each effect off.
- `Mix` and `Invert Mask`.

### Good to know

- **Lens preparation.** When an optical setting changes, the plug-in recomputes the PSFs (with a progress bar). It takes from a few tenths of a second to a few seconds, depending on the aberrations, the quality and the number of cores. Each following frame only costs the render. This computation is cached per node and for a given image size.
- **Proxy and Viewer downscale** are taken into account: the size of a pixel on the sensor and the size of the PSFs follow the render scale.
- **Anamorphic plates.** The pixel aspect of the plate is read automatically (2 for a 2× squeeze). The lens is modelled in the desqueezed space: a round bokeh stays round there and looks squeezed on the plate. `Filmback Width` is the physical, squeezed width on the sensor.
- **Image edges.** The output keeps the input format. Light that would leave the frame is lost, and the light of the edges is extended past the frame.
- **Alpha.** The source alpha is copied without blur.

### Real lenses

In the Lens tab, set `Lens Model` to **Real Lens**, then pick a lens:

| Lens | Prescription | Design field | Suggested focal length in Super 35 | What it shows |
|---|---|---|---|---|
| Double Gauss f/2 (modern Planar type) | US patent 8,879,901 | 21.6° | 75 mm and more | spherical aberration and coma wide open, fine lateral colour |
| Wide angle f/4, 100° (retrofocus) | Laikin p. 108 | 100° | 12 mm and more | strong barrel distortion, light falloff in the corners |
| Telephoto f/5.6 | Laikin p. 91 | 20° | 80 mm and more | axial colour, very even image |
| Cooke triplet f/5 | Optiland example | 40° | 40 mm and more | field curvature and astigmatism towards the edges |
| Petzval f/1.4 | Laikin p. 75 | 14° | 115 mm and more | soft centre wide open, "swirling" edges |

You can also pick `Custom File` and load your own `.lens` file.

**What the plug-in computes.** The prescription is scaled to `Focal Length`; together with `Filmback Width`, this sets the field of view. The f-number sets the diaphragm, never beyond the design aperture. For each wavelength, the plug-in traces rays through the glasses of the catalogue and derives:
- the real position of the chief ray, hence distortion and lateral colour;
- the wavefront, fitted on 45 Zernike polynomials, hence the diffraction PSF;
- the pupil that is really open: the lens mounts clip off-axis rays, which gives the cat's-eye bokeh and darker corners;
- the relative illumination.

**Focus.** The sensor is placed automatically at the best on-axis focus. `Focus Distance` (in metres, 0 = infinity) focuses on a close subject. `Defocus` (Aberrations tab) adds defocus.

**Amounts.** `Aberration Amount`, `Distortion Amount` and `Lateral Colour Amount` exaggerate or tone down the lens.

**If the image field exceeds the design field** (focal length too short for the filmback), the plug-in shows a warning. The corners are then vignetted down to black: it is the image circle of the lens.

**Diaphragm.** The shape of the diaphragm (number of blades, roundness, rotation) applies to real lenses too. The settings reserved for the parametric mode are greyed out.

#### `.lens` file format

```
name = My lens
source = where the prescription comes from
fnumber = 2.0          # design aperture
half_field = 20        # design half field (degrees)
surfaces
# radius  thickness  material  [semi-diameter|auto]  [stop]  [k=conic]  [a4=.. a6=..]
 47.07   5.30   N-LAF32
 184.3   0.60   air
 inf     8.70   air   auto stop
-17.50   1.30   SF4@hikari
 ...
 inf     3.15   air    # last thickness: up to the image plane
```

- **Lines.** One line per surface, object side first. The unit is free, since the lens is rescaled to the requested focal length.
- **Material.** `air`, a glass of the catalogue (1611 Schott, Ohara, Hoya, CDGM, Hikari and Sumita glasses; `name@vendor` when names clash), or `nd/vd` (e.g. `1.67/47.2`) for a glass given by its index and Abbe number, as in patents.
- **Semi-diameters.** When they are missing, they are computed: full aperture up to 60 % of the field, then an aperture that shrinks towards the corner. The `unvignetted_field` and `corner_pupil` keys of `carender` control this computation.
- **Bundled files.** The files of the `lenses/` folder are built into the plug-in, through `tools/gen_lenses_builtin.py`.

### Cameras and presets

**Camera sensitivities** (Spectral tab, `Camera`)

The lens separates the wavelengths; the camera decides which colour each of them takes: hue of the fringes, iridescent edges of highlights. By default, the plug-in uses the human eye (CIE 1931 observer). With a camera:
- the plate colours are taken as the camera's, brought into the working space by a 3 × 3 matrix;
- this matrix is fitted on 190 reference colours, preserving white;
- colours without any lens effect do not change (the round trip stays exact);
- only the colours of the separated wavelengths change.

The 52 built-in cameras come from **rawtoaces-data**, the Academy Software Foundation database (Apache-2.0 license): Canon, Nikon, Sony (including the A7S II and III), Fujifilm, Panasonic, Hasselblad, ARRI D21, etc. Current cinema cameras (ARRI Alexa, RED, Sony Venice) have no public data under a usable license.

For another camera, `Custom File` reads:
- either a text file, one line per wavelength: `nm r g b` (at least from 420 to 680 nm; space, tab, comma or semicolon separators);
- or a JSON file in the rawtoaces-data format.

Measurements (`test_render`): matrices fitted within 0.9 % of white on average, 1.4 % at worst (Fujifilm X-T4); round trip exact to 3·10⁻⁸ for the 52 cameras; on an isolated wavelength, the Sony A7S III sees a colour up to 0.42 away in chromaticity from the CIE observer.

**The effect is subtle, often invisible.** On the test chart with the Vintage preset, switching from the CIE observer to the Sony A7S III changes the colour of the fringes by 0.4 % (median) and 0.8 % for the 5 % most affected pixels, below the visibility threshold (about 1 to 2 %). A fringe is not an isolated wavelength but a band of wavelengths, and the matrix of each camera is fitted to render broad-spectrum light correctly, so the differences between cameras cancel out almost entirely. The effect only becomes visible with narrow-spectrum sources: saturated LEDs, lasers, sodium lamps. This setting is about fidelity, not look.

**Presets** (Lens tab, `Preset`)

A preset first resets the optical settings of the Lens and Aberrations tabs, then applies its own. Everything can be changed afterwards. The filmback, the optical centre and the settings of the Spectral and Render tabs are kept.

| Preset | Effect |
|---|---|
| Diffraction Limited f/11 | perfect lens: only the Airy disc remains |
| Vintage (coma, astigmatism) | coma and astigmatism at the edges, lateral colour, barrel distortion, cat's eye |
| Soft Wide Open f/1.4 | spherical aberration and spherochromatism: soft glow, purple and green fringes |
| Diffraction Stars f/16 | 7 straight blades: 14-point stars |
| Anamorphic Oval Bokeh | pupil twice as tall as it is wide, slight defocus |
| Real: Double Gauss f/2, Petzval f/1.4, Cooke Triplet f/5, Wide Angle f/4, Telephoto f/5.6 | traced real lenses |

For a real lens, the focal length is chosen so that the corner of the image falls at 95 % of the field the prescription was designed for, taking the filmback and the plate format into account.

**Your own presets (files)**

After the 10 built-in presets, the `Preset` menu adds the `.txt` files found in:
- the plug-in's `presets` folder (`ChromaAberrations_OFX.ofx.bundle/Contents/Resources/presets`), which holds two examples. It is replaced at every reinstallation: do not put your own files there;
- your user folder:
  - Windows: `%APPDATA%\ChromaAberrations_OFX\presets`;
  - Linux: `~/.config/ChromaAberrations_OFX/presets`;
- the folders listed in the `CA_PRESET_PATH` environment variable, separated by `;` (or `:` on Linux). Handy to share presets across a studio, on a network folder.

The menu is built when the plug-in loads: **restart Nuke after adding a file**.

The format is the one of the `carender` files, so one file gives the same lens in both tools:

```
name = My lens               # name in the menu (otherwise: the file name)
hint = Description shown in the menu tooltip
lens = petzval_f14           # real lens: built-in, or a .lens file (path relative to the preset)
fnumber = 2
focus_distance = 2           # metres, 0 = infinity
focal = 85                   # optional for a real lens: computed otherwise
camera = Sony_ILCE-7SM3      # optional: otherwise the node's camera does not change
```

Recognised keys:
- parametric lens (no `lens` key): `fnumber`, `w040`, `w131`, `w222`, `w220`, `w311`, `w020`, `axial_ca`, `lateral_ca`, `spherochromatism`, `zernike`, `focal`;
- real lens: `lens`, `focal`, `fnumber`, `focus_distance`, `aberration_gain`, `distortion_gain`, `lateral_gain`;
- both: `blades`, `roundness`, `rotation`, `aspect`, `vignetting`, `vignetting_start`, `natural_vignetting`, `camera`.

The other `carender` keys (`kernel`, `grid_x`, `wavelengths`, `output`…) are ignored: the render settings stay those of the node. A missing key takes the `carender` default (no aberrations, f/2.8, round diaphragm). Camera names are the ones `carender … camera=list` prints.

You can also keep settings with Nuke's node presets (right-click on the node → Presets). They are specific to Nuke, whereas the files also work with `carender` and other OFX hosts.

### How it works

The computation has two stages.

**1. Lens preparation.** It only depends on the settings and the image size, so it is done once and cached. It computes:
- the PSFs on the grid, for each wavelength, centred on the chief ray;
- a map of the chief-ray displacement per wavelength (distortion and lateral colour);
- a relative illumination map (cos⁴ and mechanical vignetting);
- the RGB → spectrum conversion table.

For a real lens, at most 9 wavelengths are traced, evenly spaced in 1/λ². The others are interpolated in 1/λ² (Lagrange on the 4 closest), which follows the dispersion of the glasses. On the double Gauss, 24 wavelengths computed from 9 traced ones give the chief ray within 0.003 µm and the corner PSF within 0.13 %.

**2. Rendering each frame**, for each wavelength:
- **Light of the wavelength.** Each pixel is converted to a spectrum. It is read where its chief ray comes from, which applies the inverse distortion, then multiplied by the relative illumination.
- **Variable blur.** The light arriving at a point is spread by the PSF interpolated between the 4 neighbouring grid points. This decomposes exactly into one convolution per grid point, on a tile of 2 × 2 cells weighted by the bilinear interpolation. Each convolution is done by FFT, and the tiles are summed.
- **Back to RGB.** The wavelengths are summed with the CIE 1931 colour matching functions, expressed in the working space. The sum is done in the Fourier domain, so only 3 inverse transforms are needed per tile.

On the GPU, the transforms of a tile are batched for all wavelengths (batched cuFFT). The wavelengths are processed in packs that fit in half of the free GPU memory, so Final mode also works on 6 to 8 GB cards, even in 4K. The PSF spectra are kept on the card from one frame to the next if they fit in a quarter of the free memory. Nuke's image is copied straight into page-locked transfer memory, without an intermediate copy.

Two design choices differ from the textbook methods:

1. **Variable blur: exact tiles instead of basis PSFs (PCA).** A basis-PSF decomposition was written and measured: 61.7 dB with 6 basis PSFs, 71 dB with 12. But a truncated decomposition gives slightly negative PSF values: on a highlight 200 times brighter than white, that makes black dots next to the spot. The tile method reproduces the PSF interpolation exactly (167 dB, float precision), never produces negative light, and costs about the same. Quality then only depends on the grid density, which is why the grid is the Preview / Final setting: PSF error of 3.7 % in 7 × 5 and 0.7 % in 15 × 11 on the vintage test lens.
2. **Spectrum: smoothest spectrum per colour, rather than Jakob and Hanika's method.** Each colour gets the smoothest spectrum that has exactly its RGB values and no negative light. These spectra are precomputed in a chromaticity table. The RGB → spectrum → RGB round trip is exact by construction (error 4·10⁻⁸), HDR and negative values included. Greys become flat spectra, and a saturated light becomes a narrow, positive spectrum. Same goal as Jakob and Hanika, with an exact round trip and no external table to license.

### Profiling and timings

Set the environment variable `CA_PROFILE=1` before starting Nuke. Each frame then writes a line to the console (Windows: start Nuke from a command prompt; Linux: the terminal), with the number of wavelengths, the grid and the PSF size; the lens preparation ("cached" when it comes from the cache), split into lens, PSFs and maps; the waiting time for the input image; the image copy; the GPU render (transfers, compute, number of packs) or CPU render; the output writing and the total. For example:

```
ChromaAberrations_OFX profile: frame 1, 1920x1080, image, 24 wavelengths, grid 7x5 (auto), PSF 13 px |
lens plan computed 957 ms (lens 24, PSFs 874, maps 28) | pack 27.4 ms | CPU 3549.5 ms | output 21.3 ms | total 4555.3 ms
```

Measurements in Natron, 1920 × 1080, on a 2-core CPU:

| Case | Lens preparation | Frame render (CPU) |
|---|---|---|
| Parametric, Preview (8 λ, 7 × 5, auto PSF 13 px) | 0.36 s | 1.2 s |
| Parametric, Final (24 λ, auto grid 7 × 5) | 0.96 s | 3.5 s |
| Real double Gauss, Final (24 λ, auto grid 15 × 23, auto PSF 13 px) | 18.4 s | 3.3 s |
| Real double Gauss, Final, fixed grid 15 × 11, PSF 129 px | 17.7 s | 5.5 s |

On an RTX 3060 Ti (before the latest transfer optimisations), a following 2K frame took 40 ms: 18.5 ms of GPU compute and 21.9 ms of transfers. `test_render` and `CA_PROFILE` print the figures for your machine.

**The Final preparation of a real lens is long**: 15 to 30 s on 2 cores, about 3 to 6 s on a 12- to 16-core workstation. It only happens once per optical setting; the following frames only cost the render. Interactively, `Preview in Viewer, Final in Render` avoids waiting for it. That is why the PSFs are not computed on the GPU: it would only speed up the first Final render.

### Validation

`test_psf` (24 checks) checks the PSF engine: Airy disc, energy, Seidel, diffraction stars, Zernike and distortion.

`test_render` (26 checks without GPU, 34 with) checks the renderer:

| Test | Reference | Result |
|---|---|---|
| Spectral round trip (3 spaces, 3 to 32 wavelengths, 20,000 colours) | Identity | Error ≤ 4·10⁻⁸ |
| Spectra of saturated ACEScg colours (red neon, blue LED, sodium, magenta) | No negative light | ≥ −0.8 % of the peak (a linear basis goes down to −7 %) |
| Render with every effect off | Input image | Relative error ≤ 4·10⁻⁷ |
| FFT tiles | Exact pixel-by-pixel convolution | 167 dB |
| PSF between grid points | PSF computed exactly at that point | Error of 12 % (4 × 3 grid), 3.7 % (7 × 5), 0.7 % (15 × 11) |
| Light point on a grid point | PSF of the bank | Difference 7·10⁻⁸ |
| Energy conservation | Sum of each channel | Difference 7·10⁻⁷ |
| Distortion and lateral colour | Chief-ray position per wavelength | 0.004 px |
| Natural falloff and mechanical vignetting | cos⁴; intersection area of the two circles | Difference 0.0006 |
| GPU vs CPU (with and without PSF cache, with and without blur) | CPU render | > 90 dB expected; not yet measured on an NVIDIA card |

`test_lens` (20 checks) compares the ray tracing with **Optiland**, an independent ray tracer (MIT license). The reference data are in `tests/reference` and are regenerated by `make_optiland_reference.py`.

| Test | Result |
|---|---|
| Indices of the 20 glasses used, from 400 to 700 nm | Difference < 5·10⁻⁸ |
| Paraxial focal length of the 5 lenses | Relative difference 3·10⁻¹⁰ |
| 60 chief rays, 3 wavelengths × 4 fields × 5 lenses | Difference 2·10⁻⁶ pixel in 2K |
| 294 rays through the pupil | Difference 0.03 pixel in 2K |
| Fitted wavefront: does its gradient predict the traced rays? | Within 2.3 % at worst (Petzval) |
| Render: displacement in the corner, illumination, light conservation, PSF on axis and at the edge | Pass |
| 24 wavelengths interpolated from 9 traced ones | Chief ray within 0.003 µm, PSF within 0.13 % |
| Automatic PSF size | 9 px instead of 129, all the light kept |
| Automatic grid (2 % tolerance) | 3 × 3 → 6 × 6, error 1.9 % |

The double Gauss rendered in Natron by the plug-in is identical to `carender` (120 dB). Visual tests use the `pattern:chart` chart: it shows coma comets, colour fringes around HDR points and the stretching of the Siemens stars at the edge of the field well.

Not tested yet: the GPU path on the latest version, and the plug-in in Nuke.

### Command-line tools

**`carender`: applies the lens to an image**

```
build\carender.exe my_plate.exr result.exr examples\vintage_coma.txt
build\carender.exe my_plate.exr result.exr examples\vintage_coma.txt fnumber=2.8 wavelengths=16
build\carender.exe pattern:chart chart.exr examples\vintage_coma.txt save_input=chart_source.exr
```

- **Images**: input and output are EXRs in linear light. The working space is ACEScg by default, otherwise `colorspace=rec709` or `colorspace=rec2020`. HDR and negative values are accepted. Alpha is copied as is, without blur.
- **Charts**: `pattern:chart` generates a 2048 × 1152 chart with Siemens stars and a grid of white, red and blue HDR light points. `pattern:points` and `pattern:siemens` are available too, and `width=` / `height=` change the size.

The lens settings are those of `psfmap` (list with `carender --help`). For a real lens: `lens=double_gauss_f2` (or a `.lens` file) with `focal=85`, `fnumber=2.8` and `focus_distance=3`. `camera=<id>` or `camera=<file>` picks a camera; `camera=list` prints the built-in ones. Additional keys:

| Key | Role |
|---|---|
| `wavelengths=8` | Number of wavelengths between 400 and 700 nm (8 for preview, 16 to 32 for final) |
| `grid_x=7 grid_y=5` | Grid of computed PSFs; the PSF is interpolated between these points |
| `kernel=65` | Maximum PSF size in pixels; a warning is shown if a PSF overflows |
| `focal=35`, `natural_vignetting=1` | Focal length (mm) for the natural cos⁴ falloff, and its amount (0 to 1) |
| `vignetting=0`, `vignetting_start=0.5` | Mechanical vignetting (cat's-eye pupil): depth of the cut in the corner, and where in the field it starts |
| `blur=1 distortion=1 vignette=1` | Each effect can be turned off (0) |
| `device=auto\|cpu\|gpu` | Where to compute |
| `compare=1` | Computes on GPU and CPU and prints the difference |
| `exact=1` | Also computes the exact pixel-by-pixel convolution and prints the PSNR. Slow: for small images only. |

`carender` prints the lens preparation time and the render time. On the GPU, it also prints the time of a following frame, which is the real per-frame cost in the plug-in, since the PSF spectra then stay on the card.

**`psfmap`: shows the PSFs**

```
cd examples
..\build\psfmap.exe vintage_coma.txt
```

`psfmap` writes one cell per grid point, the centre of the image in the middle, plus a CSV report (method, energy, EE80, centroid, displacement). `psfmap --help` lists the keys. The `examples/` folder holds five example lenses (`perfect`, `vintage_coma`, `spherical_f14`, `diaphragm_star`, `anamorphic`) usable by both tools.

**Lens keys** (also used by presets). The coefficients are in **waves** (at `lambda0` = 550 nm), at the edge of the pupil and in the corner of the image:

| Key | Aberration | Visible effect |
|---|---|---|
| `w040` | Spherical | Soft glow around highlights |
| `w131` | Coma | Comet-shaped spots towards the edges. Positive: tail towards the centre; negative: outwards. |
| `w222` | Astigmatism | Spots stretched radially or tangentially towards the edges |
| `w220` | Field curvature | Blur increasing towards the edges |
| `w311` | Distortion | Negative: barrel; positive: pincushion |
| `w020` | Defocus | Uniform blur that reveals the shape of the diaphragm |
| `axial_ca` | Axial colour | Purple or green fringes around sharp areas |
| `lateral_ca` | Lateral colour | Radial colour fringes towards the edges |
| `spherochromatism` | Spherical aberration varying with wavelength | Coloured glow |
| `zernike=j:c,...` | Zernike terms (Noll index ≥ 4, in waves RMS) | Free or measured aberrations |

- **Diaphragm**: `blades` (number of blades, 0 = round), `roundness` (0 for straight blades, 1 for a round diaphragm), `rotation` (degrees), `aspect` (pupil height / width, 2 = 2× anamorphic), `vignetting` (mechanical vignetting).
- **Sensor**: `filmback` (width covered by the image in mm, 24.89 = Super 35), `center_x`, `center_y` (optical centre).

### Building notes

ChromaAberrations_OFX builds like the other plug-ins (see [Building](#building)), with a few differences:

- `build.bat` and `build.sh` build the plug-in, `psfmap` and `carender`, then run `test_psf`, `test_render` and `test_lens`. On a machine with an NVIDIA card, `test_render` compares the GPU with the CPU and prints the 2K timings. When `build.bat` is run as administrator, it offers to install the bundle into the OFX folder.
- CMake options: `CA_WITH_CUDA` (`ON`), `CA_BUILD_PLUGIN` (`ON`; turn it off to build only the tools and tests, without downloading the OpenFX headers), `CMAKE_CUDA_ARCHITECTURES` (default `"75-real;86-real;89-real;75-virtual"`; the build scripts add the installed GPU if it is newer), `OFX_DIR` for a local OpenFX checkout.
- **cuFFT**: on Windows, the cuFFT DLL (`cufft64_*.dll`) is shipped in the bundle, next to the `.ofx`, and the plug-in loads it from its own folder; without it, or without a GPU, it runs on the CPU. On Linux, cuFFT is linked statically.
- The node is in the **Filter** menu.

### Files

```
src/core/      engine: spectrum, diaphragm, wavefront, PSF, PSF bank, RGB <-> spectrum, lens plan, CPU renderer
src/lens/      real lenses: glass catalogue, .lens files, ray tracing, wavefront
src/cuda/      CUDA renderer (cuFFT)
src/ofx/       OpenFX plug-in
lenses/        the 5 prescriptions built into the plug-in (.lens format)
presets/       example preset files, copied into the bundle
examples/      5 example lenses for psfmap and carender
tools/         psfmap, carender and the generators of the built-in data
tests/         test_psf, test_render, test_lens (Optiland references in tests/reference)
third_party/   pocketfft (CPU FFT), tinyexr + miniz (EXR files), rawtoaces-data license
```

A real anamorphic lens prescription is still missing (no usable public prescription found yet); anamorphic plates are handled with the parametric oval pupil.

---

## Performance

Render time per frame, CPU vs GPU, measured with `ofx_bench` (best of 5 runs). GPU times include the host ↔ GPU transfers, as in Nuke. CPU and GPU renders are identical in every test (differences below 1e-5 of the pixel values).

Two test machines:

- **Workstation A** (2010): 2 × Xeon X5675, GeForce RTX 3060 Ti, PCI Express 2.0, DDR3.
- **Workstation B** (2016): 2 × Xeon E5-2680 v4, RTX A4000, PCI Express 3.0, DDR4.

### Workstation A

| | |
|---|---|
| CPU | 2 × Intel Xeon X5675 @ 3.07 GHz, 12 cores / 24 threads in total (Westmere, no AVX2: plug-ins built without AVX2) |
| RAM | 96 GB DDR3 |
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB |
| GPU link | PCI Express 2.0, about 6 GB/s host ↔ GPU (page-locked memory) |
| System | Windows, Visual Studio 2026, CUDA Toolkit 13 |

**HD (1920×1080)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 83 ms | 22.8 ms | ×3.6 |
| push_IT Stretch | 90 ms | 26.6 ms | ×3.4 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 140 ms | 30.0 ms | ×4.7 |
| push_IT Vector Warp 300 px | 153 ms | 25.8 ms | ×5.9 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 131 ms | 24.8 ms | ×5.3 |
| push_IT Interpolate + Spread Blur 500 | 139 ms | 26.9 ms | ×5.2 |
| apGlow Max Size 2000, gaussian, 10 steps | 144 ms | 22.8 ms | ×6.3 |
| apGlow Max Size 2000, box, 10 steps | 104 ms | 21.8 ms | ×4.8 |
| apGlow Max Size 300, gaussian, 20 steps | 337 ms | 30.1 ms | ×11.2 |
| L_Despill default (log, average) | 31 ms | 20.1 ms | ×1.5 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 221 ms | 39.5 ms | ×5.6 |
| despill_IT default | 198 ms | 25.8 ms | ×7.7 |
| despill_IT + Spill Light (BG) + Light Wrap | 245 ms | 34.3 ms | ×7.1 |

**UHD (3840×2160)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 252 ms | 84.7 ms | ×3.0 |
| push_IT Stretch | 275 ms | 89.2 ms | ×3.1 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 492 ms | 104.6 ms | ×4.7 |
| push_IT Vector Warp 300 px | 444 ms | 100.8 ms | ×4.4 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 339 ms | 95.9 ms | ×3.5 |
| push_IT Interpolate + Spread Blur 500 | 461 ms | 96.6 ms | ×4.8 |
| apGlow Max Size 2000, gaussian, 10 steps | 620 ms | 89.6 ms | ×6.9 |
| apGlow Max Size 2000, box, 10 steps | 358 ms | 92.1 ms | ×3.9 |
| apGlow Max Size 300, gaussian, 20 steps | 1583 ms | 112.9 ms | ×14.0 |
| L_Despill default (log, average) | 146 ms | 77.9 ms | ×1.9 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 828 ms | 146.2 ms | ×5.7 |
| despill_IT default | 606 ms | 93.2 ms | ×6.5 |
| despill_IT + Spill Light (BG) + Light Wrap | 713 ms | 124.7 ms | ×5.7 |

### Workstation B

| | |
|---|---|
| CPU | 2 × Intel Xeon E5-2680 v4 @ 2.40 GHz, 28 cores / 56 threads in total (Broadwell) |
| RAM | 128 GB DDR4 |
| GPU | NVIDIA RTX A4000, 16 GB |
| GPU link | PCI Express 3.0, about 12 GB/s host ↔ GPU (page-locked memory) |
| System | Windows 10, Visual Studio 2026, CUDA Toolkit 13 |

**HD (1920×1080)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 82 ms | 9.3 ms | ×8.8 |
| push_IT Stretch | 80 ms | 13.4 ms | ×5.9 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 115 ms | 16.9 ms | ×6.8 |
| push_IT Vector Warp 300 px | 134 ms | 13.6 ms | ×9.9 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 118 ms | 11.0 ms | ×10.7 |
| push_IT Interpolate + Spread Blur 500 | 117 ms | 16.0 ms | ×7.3 |
| apGlow Max Size 2000, gaussian, 10 steps | 115 ms | 13.1 ms | ×8.8 |
| apGlow Max Size 2000, box, 10 steps | 95 ms | 12.7 ms | ×7.5 |
| apGlow Max Size 300, gaussian, 20 steps | 256 ms | 19.3 ms | ×13.2 |
| L_Despill default (log, average) | 22 ms | 8.5 ms | ×2.6 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 105 ms | 23.2 ms | ×4.5 |
| despill_IT default | 224 ms | 13.3 ms | ×16.8 |
| despill_IT + Spill Light (BG) + Light Wrap | 262 ms | 20.2 ms | ×13.0 |

**UHD (3840×2160)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 201 ms | 34.8 ms | ×5.8 |
| push_IT Stretch | 203 ms | 40.7 ms | ×5.0 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 313 ms | 56.6 ms | ×5.5 |
| push_IT Vector Warp 300 px | 335 ms | 43.0 ms | ×7.8 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 260 ms | 41.5 ms | ×6.3 |
| push_IT Interpolate + Spread Blur 500 | 309 ms | 46.6 ms | ×6.6 |
| apGlow Max Size 2000, gaussian, 10 steps | 295 ms | 49.4 ms | ×6.0 |
| apGlow Max Size 2000, box, 10 steps | 216 ms | 48.3 ms | ×4.5 |
| apGlow Max Size 300, gaussian, 20 steps | 606 ms | 77.5 ms | ×7.8 |
| L_Despill default (log, average) | 67 ms | 29.4 ms | ×2.3 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 330 ms | 86.5 ms | ×3.8 |
| despill_IT default | 484 ms | 48.6 ms | ×10.0 |
| despill_IT + Spill Light (BG) + Light Wrap | 557 ms | 63.7 ms | ×8.8 |

### Summary

The GPU is 1.5 to 14 times faster than the CPU on workstation A, and 2.3 to 17 times faster on workstation B. The gain grows with the cost of the settings (large blurs, distances, Vector Warp, many glow steps). Most of the GPU time is spent moving the image to and from the graphics card: 70 to 95 % on workstation A (PCI Express 2.0, about 3 GB/s in practice), still about 45 to 80 % on workstation B (PCI Express 3.0, about 9 GB/s). The GPU renders are 2 to 3 times faster on workstation B, mostly thanks to the faster bus, while the CPU renders range from slightly slower to 2.5 times faster depending on the effect: more CPU cores help less than a faster bus. A simple per-pixel effect such as the default L_Despill gains little; spatial filters (median, blurs, distances, many glow steps) gain the most. On a PC with PCI Express 4.0 or 5.0, the transfers are faster again.

### Playback in Nuke

`ofx_bench` measures the plug-ins alone. In Nuke, each frame also goes through the Read node (reading and decoding the file), the hand-over of the images between Nuke and the plug-in, and the Viewer. Measured on workstation B with despill_IT_OFX in HD (GPU), without any cache:

| Source | Read alone | Read + despill_IT_OFX |
|---|---|---|
| MP4 (H.264) | — | 10–11 fps (decoding: about 56 ms per frame) |
| EXR, 32-bit float, uncompressed | 14–16 fps | 10–11 fps |
| EXR, 16-bit half, ZIP (1 scanline) | 19 fps | about 12 fps (estimated) |

- The plug-in itself takes about 16 ms per frame (GPU, transfers included), plus 10 to 15 ms for Nuke to hand the images over: about a third of the frame time. Reading, decoding and displaying the images in Nuke cost more than the effect.
- **Prefer image sequences to video files**: an MP4 is decoded on the CPU at every frame, and its 4:2:0 chroma also blurs the edges of the screen. **EXR 16-bit half with ZIP (1 scanline) compression** reads faster than uncompressed 32-bit EXR, which needs about 800 MB/s at 24 fps in HD.
- **For real-time playback**, play the shot once to fill the Viewer cache (green bar of the timeline), or use a Flipbook: the following playbacks run in real time.
- **Profiling despill_IT_OFX in Nuke**: start Nuke with the environment variable `DESPILLIT_PROFILE=1` (Windows: `set DESPILLIT_PROFILE=1` in a command prompt, then start Nuke from it). Each render writes a line to `%TEMP%\despill_IT_profile.txt` (Linux: `/tmp/despill_IT_profile.txt`): total time, time to fetch the output and the source images, GPU or CPU path, and the time of each GPU stage. Without the variable, nothing is measured.

---

## Requirements

| | Windows 10 / 11 (x64) | Linux (Mint 22 / Ubuntu 24.04, x86-64) | macOS 11+ |
|---|---|---|---|
| Compiler | Visual Studio 2022 or 2026 (Community is fine) with the **Desktop development with C++** workload | `build-essential` (g++), `libgl-dev` | Xcode Command Line Tools |
| Build tools | [CMake](https://cmake.org/download/) ≥ 3.20 (tick *Add CMake to the system PATH*) and [Git](https://git-scm.com/download/win) | `cmake`, `git` | CMake ≥ 3.20, Git |
| GPU (optional) | NVIDIA GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x | NVIDIA driver 580+ and CUDA Toolkit 13 (NVIDIA repository for Ubuntu 24.04) | not available (CPU only) |

Notes:
- The OpenFX SDK (`OFX_Release_1.5.1`) is downloaded automatically from GitHub during the first configuration, which is why Git is required. CornerPinEx_OFX and ChromaAberrations_OFX download Natron's OFX headers and support library instead.
- Visual Studio 2026 needs CUDA Toolkit 13.x. The Nsight warnings shown by the CUDA installer about Visual Studio integration can be ignored.
- **Nuke versions**: tested with Nuke 14 and Nuke 17 on Windows; one build works for all versions. Nuke 14 and older ship their own, older Microsoft C++ runtime (`msvcp140.dll`), which plug-ins built with Visual Studio 2022 17.10 or newer would crash with. The CMake projects define `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` on Windows so that the plug-ins work with every Nuke version.
- The CUDA runtime is linked statically: the plug-ins only need a recent NVIDIA driver on the machine that runs Nuke, no CUDA DLL to ship (except the cuFFT DLL of ChromaAberrations_OFX on Windows, copied into its bundle).
- On Linux, the C++ runtime is linked statically too: the plug-ins only depend on glibc 2.38 or newer and, for push_IT_OFX, apGlow_OFX and despill_IT_OFX, on `libGL.so.1` (viewer overlays), which every machine running Nuke has.
- Supported GPUs by default: Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx), and Blackwell (RTX 50xx) with CUDA 12.8+, plus PTX for newer cards. For a Pascal card (GTX 10xx), use CUDA 12 and add `61-real` to the architectures (see the CMake options below).

---

## Building

Each plug-in is built on its own, from its folder. The commands below use `push_IT_OFX` as an example; replace it with the folder of the plug-in you want to build (e.g. `apGlow_OFX`).

### Windows

Double-click `build.bat` in the plug-in folder, or run it from a terminal:

```bat
cd push_IT_OFX
build.bat          :: CPU + CUDA (CUDA is used if the CUDA Toolkit is installed)
build.bat cpu      :: CPU only
build.bat clean    :: delete the build folder and rebuild from scratch
```

The script checks the prerequisites, picks the newest Visual Studio installed and writes everything to `build.log`. The plug-in ends up in `build\push_IT_OFX.ofx.bundle`.

Run `build.bat clean` after changing your Visual Studio or CUDA version.

### Linux

```bash
sudo apt install build-essential cmake git libgl-dev
cd push_IT_OFX
./build.sh            # CPU + CUDA (if nvcc is found, also in /usr/local/cuda/bin)
./build.sh cpu        # CPU only
./build.sh clean      # rebuild from scratch
./build.sh install    # build, then copy the bundle to /usr/OFX/Plugins (sudo)
```

### macOS (CPU only)

```bash
cd push_IT_OFX
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPUSHIT_WITH_CUDA=OFF \
      -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
cmake --build build -j$(sysctl -n hw.ncpu)
```

Each plug-in has its own CUDA option name, e.g. `-DAPGLOW_WITH_CUDA=OFF` for apGlow_OFX (see the table below).

### Manual CMake build and options

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

| Option | Default | Effect |
|---|---|---|
| `PUSHIT_WITH_CUDA` / `APGLOW_WITH_CUDA` / `LDESPILL_WITH_CUDA` / `DESPILLIT_WITH_CUDA` / `CORNERPINEX_OFX_WITH_CUDA` / `CA_WITH_CUDA` | `ON` | Build the CUDA back-end (if a CUDA compiler is found). |
| `PUSHIT_CUDA_ARCHS` / `APGLOW_CUDA_ARCHS` / `LDESPILL_CUDA_ARCHS` / `DESPILLIT_CUDA_ARCHS` | empty (automatic) | GPU architectures, e.g. `"75-real;86-real"`. |
| `APGLOW_AVX2` | `OFF` | AVX2 / FMA on the CPU, about 15 % faster. The plug-in will **not load** on a CPU without AVX2 (before Intel Haswell / AMD Ryzen). |
| `OPENFX_ROOT` | empty | Use a local OpenFX SDK checkout instead of downloading it. |

CornerPinEx_OFX and ChromaAberrations_OFX use `CMAKE_CUDA_ARCHITECTURES` and `OFX_DIR` (plus `SUPPORTEXT_DIR` for CornerPinEx_OFX) instead (see [CornerPinEx_OFX](#building-and-checking-the-installation) and [ChromaAberrations_OFX](#building-notes)).

---

## Installation in Nuke

Copy the `.ofx.bundle` folder into the OFX plug-in folder of your system:

| System | Folder |
|---|---|
| Windows | `C:\Program Files\Common Files\OFX\Plugins\` |
| Linux | `/usr/OFX/Plugins/` |
| macOS | `/Library/OFX/Plugins/` |

You can also put the bundles in any folder and add that folder to the `OFX_PLUGIN_PATH` environment variable.

The nodes appear in the **Filter** menu (push_IT_OFX, apGlow_OFX, ChromaAberrations_OFX), the **Keyer** menu (L_Despill_OFX, despill_IT_OFX) and the **Transform** menu (CornerPinEx_OFX), or with Tab.

**After an installation or an update, clear Nuke's OFX cache** (close Nuke first). Otherwise Nuke may keep showing the old version or not show the node at all:
- Windows: delete `%LOCALAPPDATA%\Temp\nuke\ofxplugincache`
- Linux / macOS: delete the `ofxplugincache` folder in Nuke's temporary directory (`NUKE_TEMP_DIR`, `/var/tmp/nuke` by default on Linux)

**Node colour**: OFX plug-ins cannot set their own colour in the Node Graph. Add a rule in *Preferences > Panels > Node Colors* with the node name (e.g. `push_IT_OFX`).

**Channels**: Nuke only passes the rgba layer to OFX plug-ins, so the effects work on rgba. Use a Shuffle before the node to process another layer.

---

## Viewer overlays

push_IT_OFX, apGlow_OFX and despill_IT_OFX draw widgets in the Viewer while their panel is open (see each plug-in). Like the handles of Nuke's own nodes (a Transform, for instance), they cannot be dragged while the Viewer is playing: pause the playback to use them. The parameters of the panel can still be changed during playback.

---

## GPU notes

Nuke does not give GPU buffers to OFX plug-ins. With *Use GPU* on, each frame is sent to the graphics card, processed entirely there, and copied back. The copies go through page-locked memory, overlapped with the transfers and spread over several CPU cores; apGlow_OFX only transfers the RGB channels it needs. In practice the host ↔ GPU transfers, not the processing, set the GPU render time, so the gain is largest on heavy settings (large blurs, Vector Warp, many glow steps).

If CUDA is not available or a GPU render fails, the frame is rendered on the CPU and a warning is written to Nuke's log.

---

## Benchmark (optional)

`ofx_bench` renders a set of test cases in HD and UHD on the CPU and on the GPU, prints the timings per stage and checks that both renders are identical. Nuke is not needed.

- Windows: run `ofx_bench\build_and_run.bat`. The results are written to `bench_result.txt`.
- Linux: run `ofx_bench/run.sh`.

The benchmark contains a copy of the plug-in sources (`ofx_bench/src/push_it`, `ofx_bench/src/expo_glow` (apGlow_OFX), `ofx_bench/src/l_despill`, `ofx_bench/src/despill_it`): copy the files again after changing a plug-in.

---

## Credits

- apGlow_OFX is based on **apGlow v1.2** by Adrian Pueyo, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022.
- push_IT_OFX is inspired by the *Pixel Spread* effect of Autodesk Flame.
- L_Despill_OFX is based on the **L_Despill v05** gizmo by Justin Johnson (Luma Pictures) and the **DespillMadness** gizmo.
- CornerPinEx_OFX is derived from **Natron's CornerPin** ([openfx-misc](https://github.com/NatronGitHub/openfx-misc) and [openfx-supportext](https://github.com/NatronGitHub/openfx-supportext), © INRIA and The Natron Developers, GPL v2+).
- ChromaAberrations_OFX uses pocketfft (BSD 3-Clause), tinyexr and miniz (BSD 3-Clause / MIT), the camera sensitivities of rawtoaces-data (Academy Software Foundation, Apache-2.0), the glass data of [refractiveindex.info](https://github.com/polyanskiy/refractiveindex.info-database) (CC0), lens prescriptions from M. Laikin's *Lens Design* and US patent 8,879,901 as published in [Optiland](https://github.com/HarrisonKramer/optiland) (MIT), which also provides the ray-tracing reference data.
- Built on the [OpenFX](https://github.com/AcademySoftwareFoundation/openfx) SDK.
