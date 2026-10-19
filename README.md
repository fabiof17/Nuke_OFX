# Nuke OFX Plug-ins

A collection of OpenFX plug-ins for The Foundry Nuke, written in C++ with optional NVIDIA CUDA acceleration.

| Plug-in | What it does |
|---|---|
| [**push_IT_OFX**](#push_it_ofx) | Edge extend / pixel spread: pushes the colours of a matte outward, or fills a matted area from its surroundings. |
| [**expo_GLOW_OFX**](#expo_glow_ofx) | Exponential glow: a sum of blurs whose sizes grow geometrically, for a soft and wide light falloff. |
| [**L_Despill_OFX**](#l_despill_ofx) | Spill suppression for green, blue and red screens, with spill matte processing, spill replace and background re-light. |
| [**despill_IT_OFX**](#despill_it_ofx) | Spatially aware despill: local screen colour, despill fading from the edges inwards, protected hues, edge colour rebuilt from the subject, background light and light wrap. |

The plug-ins run on Windows, Linux and macOS. On Windows and Linux they can render on an NVIDIA GPU with CUDA; when no GPU is available they fall back to a multithreaded CPU renderer automatically. The CPU and GPU renders are identical.

Ready-to-use builds for Windows and Linux are available on the [Releases](../../releases) page.

Each plug-in lives in its own folder and is built on its own:

```
push_IT_OFX/      edge extend plug-in (sources, CMakeLists.txt, build.bat, build.sh)
expo_GLOW_OFX/    glow plug-in (sources, CMakeLists.txt, build.bat, build.sh)
L_Despill_OFX/    classic despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
despill_IT_OFX/   spatially aware despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
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
- **Source Premultiplied**: in Expand, unpremultiplies the source so that the edges do not darken.
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

### Recipes

- **Edge extend before a despill or a merge**: premultiplied plate in Source, Expand, Interpolate, Edge Shrink 1–2, Alpha Output = Input, then premultiply.
- **Quick clean plate**: plate in Source, object roto in Matte, Contract, Interpolate or Stretch, Edge Shrink 2–5, Spread Blur to taste.
- **Directional stretch**: Vector Warp with Vector Source = Constant, angle set to the direction the pixels should flow. For local control, connect motion vectors (in pixels) to the Vector input.

---

## expo_GLOW_OFX

Exponential glow built with the multi-blur method: several blurs whose sizes grow geometrically up to *Max Size* are added together, with a colour gain that changes progressively from the smallest blur to the largest. The result imitates the soft, wide falloff of real light diffusion.

This is a C++ / CUDA re-implementation of the **apGlow** gizmo by Adrian Pueyo (apGlow v1.2, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022), calibrated to match it.

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

With the **Screen Color** colour picker, sample a clean, evenly lit part of the screen. This is the most important setting: it gives the reference hue and the channel to treat (green or blue).

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

## Performance

Render time per frame, CPU vs GPU, measured with `ofx_bench` (best of 5 runs). GPU times include the host ↔ GPU transfers, as in Nuke. CPU and GPU renders are identical in every test (differences below 1e-5 of the pixel values).

**Test configuration**

| | |
|---|---|
| CPU | Intel Xeon X5675 @ 3.07 GHz, 6 cores / 12 threads (Westmere, no AVX2: plug-ins built without AVX2) |
| RAM | 96 GB DDR3 |
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB |
| GPU link | PCI Express 2.0, about 6 GB/s host ↔ GPU (page-locked memory) |
| System | Windows, Visual Studio 2026, CUDA Toolkit 13 |

**HD (1920×1080)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 72 ms | 22.2 ms | ×3.3 |
| push_IT Stretch | 80 ms | 24.9 ms | ×3.2 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 133 ms | 29.1 ms | ×4.6 |
| push_IT Vector Warp 300 px | 155 ms | 26.0 ms | ×6.0 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 117 ms | 24.7 ms | ×4.7 |
| push_IT Interpolate + Spread Blur 500 | 126 ms | 28.5 ms | ×4.4 |
| expo_GLOW Max Size 2000, gaussian, 10 steps | 152 ms | 22.8 ms | ×6.7 |
| expo_GLOW Max Size 2000, box, 10 steps | 102 ms | 21.6 ms | ×4.7 |
| expo_GLOW Max Size 300, gaussian, 20 steps | 375 ms | 30.4 ms | ×12.3 |
| L_Despill default (log, average) | 46 ms | 21.0 ms | ×2.2 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 346 ms | 47.7 ms | ×7.2 |
| despill_IT default | 189 ms | 27.3 ms | ×6.9 |
| despill_IT + Spill Light (BG) + Light Wrap | 227 ms | 31.9 ms | ×7.1 |

**UHD (3840×2160)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 292 ms | 84.0 ms | ×3.5 |
| push_IT Stretch | 276 ms | 89.7 ms | ×3.1 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 552 ms | 101.9 ms | ×5.4 |
| push_IT Vector Warp 300 px | 492 ms | 101.9 ms | ×4.8 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 343 ms | 101.1 ms | ×3.4 |
| push_IT Interpolate + Spread Blur 500 | 555 ms | 100.2 ms | ×5.5 |
| expo_GLOW Max Size 2000, gaussian, 10 steps | 842 ms | 90.6 ms | ×9.3 |
| expo_GLOW Max Size 2000, box, 10 steps | 407 ms | 98.5 ms | ×4.1 |
| expo_GLOW Max Size 300, gaussian, 20 steps | 2209 ms | 116.1 ms | ×19.0 |
| L_Despill default (log, average) | 215 ms | 79.5 ms | ×2.7 |
| L_Despill Median 5 + Clean 4 + Matte Blur 20 + BG | 1520 ms | 155.3 ms | ×9.8 |
| despill_IT default | 640 ms | 91.9 ms | ×7.0 |
| despill_IT + Spill Light (BG) + Light Wrap | 816 ms | 123.8 ms | ×6.6 |

The GPU is 2 to 19 times faster than the CPU. The gain grows with the cost of the settings (large blurs, Vector Warp, many glow steps), because most of the GPU time is spent moving the image to and from the graphics card: 70 to 95 % in these tests. A simple per-pixel effect such as the default L_Despill gains little; spatial filters (median, blurs, distances, many glow steps) gain the most. This test machine is a 2010-era workstation with PCI Express 2.0: on a recent PC (PCI Express 4.0 or newer, DDR4 / DDR5) the transfers are much faster, so the GPU times should be lower and the speed-ups higher.

---

## Requirements

| | Windows 10 / 11 (x64) | Linux (Mint 22 / Ubuntu 24.04, x86-64) | macOS 11+ |
|---|---|---|---|
| Compiler | Visual Studio 2022 or 2026 (Community is fine) with the **Desktop development with C++** workload | `build-essential` (g++) | Xcode Command Line Tools |
| Build tools | [CMake](https://cmake.org/download/) ≥ 3.20 (tick *Add CMake to the system PATH*) and [Git](https://git-scm.com/download/win) | `cmake`, `git` | CMake ≥ 3.20, Git |
| GPU (optional) | NVIDIA GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.x or 13.x | NVIDIA driver 580+ and CUDA Toolkit 13 (NVIDIA repository for Ubuntu 24.04) | not available (CPU only) |

Notes:
- The OpenFX SDK (`OFX_Release_1.5.1`) is downloaded automatically from GitHub during the first configuration, which is why Git is required.
- Visual Studio 2026 needs CUDA Toolkit 13.x. The Nsight warnings shown by the CUDA installer about Visual Studio integration can be ignored.
- The CUDA runtime is linked statically: the plug-ins only need a recent NVIDIA driver on the machine that runs Nuke, no CUDA DLL to ship.
- On Linux, the C++ runtime is linked statically too: the plug-ins only depend on glibc 2.38 or newer.
- Supported GPUs by default: Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx), and Blackwell (RTX 50xx) with CUDA 12.8+, plus PTX for newer cards. For a Pascal card (GTX 10xx), use CUDA 12 and add `61-real` to the architectures (see the CMake options below).

---

## Building

Each plug-in is built on its own, from its folder. The commands below use `push_IT_OFX` as an example; replace it with the folder of the plug-in you want to build (e.g. `expo_GLOW_OFX`).

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
sudo apt install build-essential cmake git
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

Each plug-in has its own CUDA option name, e.g. `-DEXPOGLOW_WITH_CUDA=OFF` for expo_GLOW_OFX (see the table below).

### Manual CMake build and options

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

| Option | Default | Effect |
|---|---|---|
| `PUSHIT_WITH_CUDA` / `EXPOGLOW_WITH_CUDA` / `LDESPILL_WITH_CUDA` / `DESPILLIT_WITH_CUDA` | `ON` | Build the CUDA back-end (if a CUDA compiler is found). |
| `PUSHIT_CUDA_ARCHS` / `EXPOGLOW_CUDA_ARCHS` / `LDESPILL_CUDA_ARCHS` / `DESPILLIT_CUDA_ARCHS` | empty (automatic) | GPU architectures, e.g. `"75-real;86-real"`. |
| `EXPOGLOW_AVX2` | `OFF` | AVX2 / FMA on the CPU, about 15 % faster. The plug-in will **not load** on a CPU without AVX2 (before Intel Haswell / AMD Ryzen). |
| `OPENFX_ROOT` | empty | Use a local OpenFX SDK checkout instead of downloading it. |

---

## Installation in Nuke

Copy the `.ofx.bundle` folder into the OFX plug-in folder of your system:

| System | Folder |
|---|---|
| Windows | `C:\Program Files\Common Files\OFX\Plugins\` |
| Linux | `/usr/OFX/Plugins/` |
| macOS | `/Library/OFX/Plugins/` |

You can also put the bundles in any folder and add that folder to the `OFX_PLUGIN_PATH` environment variable.

The nodes appear in the **Filter** menu (push_IT_OFX, expo_GLOW_OFX) and the **Keyer** menu (L_Despill_OFX, despill_IT_OFX), or with Tab.

**After an installation or an update, clear Nuke's OFX cache** (close Nuke first). Otherwise Nuke may keep showing the old version or not show the node at all:
- Windows: delete `%LOCALAPPDATA%\Temp\nuke\ofxplugincache`
- Linux / macOS: delete the `ofxplugincache` folder in Nuke's temporary directory (`NUKE_TEMP_DIR`, `/var/tmp/nuke` by default on Linux)

**Node colour**: OFX plug-ins cannot set their own colour in the Node Graph. Add a rule in *Preferences > Panels > Node Colors* with the node name (e.g. `push_IT_OFX`).

**Channels**: Nuke only passes the rgba layer to OFX plug-ins, so the effects work on rgba. Use a Shuffle before the node to process another layer.

---

## GPU notes

Nuke does not give GPU buffers to OFX plug-ins. With *Use GPU* on, each frame is sent to the graphics card, processed entirely there, and copied back. The copies go through page-locked memory, overlapped with the transfers and spread over several CPU cores; expo_GLOW_OFX only transfers the RGB channels it needs. In practice the host ↔ GPU transfers, not the processing, set the GPU render time, so the gain is largest on heavy settings (large blurs, Vector Warp, many glow steps).

If CUDA is not available or a GPU render fails, the frame is rendered on the CPU and a warning is written to Nuke's log.

---

## Benchmark (optional)

`ofx_bench` renders a set of test cases in HD and UHD on the CPU and on the GPU, prints the timings per stage and checks that both renders are identical. Nuke is not needed.

- Windows: run `ofx_bench\build_and_run.bat`. The results are written to `bench_result.txt`.
- Linux: run `ofx_bench/run.sh`.

The benchmark contains a copy of the plug-in sources (`ofx_bench/src/push_it`, `ofx_bench/src/expo_glow`, `ofx_bench/src/l_despill`, `ofx_bench/src/despill_it`): copy the files again after changing a plug-in.

---

## Credits

- expo_GLOW_OFX is based on **apGlow v1.2** by Adrian Pueyo, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022.
- push_IT_OFX is inspired by the *Pixel Spread* effect of Autodesk Flame.
- L_Despill_OFX is based on the **L_Despill v05** gizmo by Justin Johnson (Luma Pictures) and the **DespillMadness** gizmo.
- Built on the [OpenFX](https://github.com/AcademySoftwareFoundation/openfx) SDK.
