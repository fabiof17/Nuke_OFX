# Nuke OFX Plug-ins

A collection of OpenFX plug-ins for The Foundry Nuke, written in C++ with optional NVIDIA CUDA acceleration.

| Plug-in | What it does |
|---|---|
| [**push_IT_OFX**](#push_it_ofx) | Edge extend / pixel spread: pushes the colours of a matte outward, or fills a matted area from its surroundings. |
| [**apGlow_OFX**](#apglow_ofx) | Exponential glow: a sum of blurs whose sizes grow geometrically, for a soft and wide light falloff. |
| [**L_Despill_OFX**](#l_despill_ofx) | Spill suppression for green, blue and red screens, with spill matte processing, spill replace and background re-light. |
| [**despill_IT_OFX**](#despill_it_ofx) | Spatially aware despill: local screen colour, despill fading from the edges inwards, protected hues, edge colour rebuilt from the subject, background light and light wrap. |
| [**MasterKeyer_OFX**](#masterkeyer_ofx) | Complete keyer inspired by the Master Keyer of Autodesk Flame: matte from two screen samples, colour patches, degrain, automatic spill suppression, edge blend and composite. |

The plug-ins run on Windows, Linux and macOS. On Windows and Linux they can render on an NVIDIA GPU with CUDA; when no GPU is available they fall back to a multithreaded CPU renderer automatically. The CPU and GPU renders are identical.

Ready-to-use builds for Windows and Linux are available on the [Releases](../../releases) page.

Each plug-in lives in its own folder and is built on its own:

```
push_IT_OFX/      edge extend plug-in (sources, CMakeLists.txt, build.bat, build.sh)
apGlow_OFX/       glow plug-in (sources, CMakeLists.txt, build.bat, build.sh)
L_Despill_OFX/    classic despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
despill_IT_OFX/   spatially aware despill plug-in (sources, CMakeLists.txt, build.bat, build.sh)
MasterKeyer_OFX/  keyer plug-in (sources, CMakeLists.txt, build.bat, build.sh)
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
- **Show Overlay**: while the node panel is open, the viewer shows:
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
- **Show Overlay**: while the node panel is open, the viewer shows one circle per glow step (diameter = blur size of the step, *Size Scale* included), the largest one solid. Drag its handle to set *Max Size*. The widget stays at the centre of the format.

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
- **Show Overlay**: shows the sample area and, with *Spatial* on, a **distance probe**: a solid circle of radius *Edge Width* and a dashed one of radius *Edge Width + Edge Falloff*. It stays at the centre of the format; drag its handles to change the two values.

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

## MasterKeyer_OFX

A complete keyer in one node, inspired by the workflow of the **Master Keyer** of Autodesk Flame: pick the screen, refine the matte, fix problem colours with patches, degrain, and get a despilled, edge-blended foreground composited over the background.

> **Not a copy of Flame.** Autodesk has not published the Master Keyer algorithm. MasterKeyer_OFX is an independent implementation that follows the same workflow and offers similar controls; its maths are its own, and its results differ from Flame's.

How it works:

- **Matte**: each pixel gets the fraction of screen colour it contains: the screen channel against a limit made of the two other channels, compared with the same measure on the screen colour. 1 = as much screen as the screen itself (transparent), 0 = no screen (opaque). The **Secondary** sample and **Mix** set the screen level that must be fully transparent, so that darker parts of the screen are cleaned too. Gain (with its Shadows / Midtones / Highlights corrections), clips, softness, shrink / grow, blur, Garbage and Holdout inputs follow. The matte is computed on the **Key In** input when it is connected (as in Flame), so that a denoised or graded version of the plate can be keyed while the original is kept in the result.
- **Matte Mode**: the matte above is a *colour-difference* matte (family of Nuke's IBK or Keylight): accurate on semi-transparent edges. The **3D** mode works like Flame's 3D Keyer: every colour is a point of a 3D space (luma, Cb, Cr); the Primary and Secondary samples define a *tolerance volume* that becomes transparent, surrounded by a *softness envelope* where the matte goes from 0 to 1 (Gain and Shadows / Midtones / Highlights widen or narrow it). It isolates any colour precisely and keeps colours far from the screen solid. **Combined** takes the larger of the two mattes: the 3D matte keeps the inside of the subject solid, the difference matte keeps the edges.
- **Local screen**: optionally, the screen colour is estimated around each pixel from the screen areas of the plate (pixels whose colour is close to the Primary) and interpolated behind the subject, which handles uneven screens.
- **Patches**: three colour patches force a set of colours to the transparent (Black) or opaque (White) part of the matte, or erode the semi-transparent edges where they are (Edge Analysis). Each patch holds a range of colours that grows with every sample added.
- **Degrain**: the colour used to compute the matte (not the image itself) is smoothed, so grain and compression noise do not make the matte noisy.
- **Spill**: automatic colour suppression relative to the screen colour, full at the edges of the subject and fading towards its inside (exact distance to the screen), so that colours close to the screen deep inside the subject are kept. A separate **Spill Colour** changes the suppression without changing the key.
- **Blend**: over a band inside the edge of the subject, the luma can be lowered or raised and the colour pushed towards a chosen colour or towards the colour of the blurred background, to integrate the edges.

### Inputs

| Input | Role |
|---|---|
| **Source** | Green / blue screen plate (Flame's *Front*): used for the result. |
| **Key In** | Optional. Plate used to compute the matte (Flame's *Key In*), e.g. a denoised or graded version of the Source. Without it, the matte is computed on the Source. |
| **BG** | Optional. Background: the result is composited over it, and its blurred colour can be used for the edge blend. |
| **Garbage** | Optional. Matte of areas forced transparent (alpha, or luminance of an RGB image). |
| **Holdout** | Optional. Matte of areas forced opaque. |

### Parameters

The parameters are split into five tabs. Each section has a **Reset** button, as in Flame.

**Matte**
- **Use GPU**: renders on the NVIDIA GPU. The detected card is shown on the right.
- **Output**: Result (foreground over BG; without BG, identical to Premult), Premult (despilled foreground premultiplied by the matte, never composited: merge it later while keeping the BG connected for *From BG*), Source CC (the despilled plate only, without the blend, not premultiplied), Alpha (the matte), Spill (amount removed), Edge (blend region).
- **Primary**: screen colour, picked on the screen. Its largest component gives the screen channel. As in Flame, the first pick also sets the Secondary and turns Auto CC on, and the Spill Colour follows the Primary until it is changed.
- **Pick Colour** (next to Primary, Secondary and each patch colour): picks the colour on the **Key In** input if it is connected, else on the **Source** (the Spill Colour is always picked on the Source), never on the keyed output. Press it, then click in the viewer for one pixel, or drag for the average of an area. Press it again or Esc to cancel. **Undo Pick** (Matte and Patches tabs) undoes the last picks, up to 20 (Nuke's Ctrl+Z does not undo the colours a pick sets, because the effect sets them itself after reading the image). These colours are plain r, g, b fields, without Nuke's colour swatch: Nuke's own picker reads the output of the node, which is keyed and premultiplied, so it would give wrong colours here.
- **Secondary** and **Mix**: second sample, typically on the darkest part of the screen. At Mix 0 only screen as strong as the Primary is fully transparent; at Mix 1 everything at least as "screen" as the Secondary is.
- **Screen**: Samples (the Primary everywhere) or Local (screen estimated around each pixel; in 3D, the tolerance volume follows it).
- **Matte Mode**: Difference, 3D or Combined (see above).
- **Tolerance** (3D): radius of the tolerance volume around the samples (Primary, and Secondary as far as Mix goes).
- **3D Softness** (3D): width of the softness envelope.
- **Luma Weight** (3D): weight of the luminance in the colour distance. Lower: lighting changes of the screen matter less.
- **Show 3D View** and **Corner** (3D, Combined): while the panel is open, a 3D view of the colour space is drawn in a corner of the viewer: the axes (luma × Luma Weight up in white, Cb in blue, Cr in red; dashed: neutral greys), the Primary and Secondary samples in their own colours, the **tolerance volume** (yellow) and the **softness envelope** (cyan, where the matte reaches 1), deformed by Gain and Shadows / Midtones / Highlights. Drag in it to rotate, Shift + drag to zoom, double-click to reset the view; drag the small triangle of its inner corner to resize it. The volumes are cut at the black level (no colour is darker than black), shown as a floor grid. **View Space** chooses the space of the view (display only): *Y Cb Cr*, the space in which the 3D matte measures its distances, or *RGB*, the RGB cube (R, G, B axes from black, greys on the dashed diagonal, the whole cube framed), where the same volumes appear tilted and stretched along the grey diagonal. **Tolerance Display** and **Softness Display** show each volume as a wireframe or as a Gouraud-shaded surface (the envelope is then translucent, so that the tolerance volume stays visible inside it). It follows the settings live (in Local screen mode it shows the volume of the Primary).
- **Balance** (Difference, Combined): weight between the two other channels when measuring the screen. For a green screen, -1 uses red only, +1 blue only.
- **Gain**: overall softness of the matte (higher: more transparency).
- **Shadows**, **Midtones**, **Highlights**: the same softness in the dark, mid and bright parts of the image (by luma, as Flame's matte sliders). Each unit doubles (positive) or halves (negative) the Gain there. The three ranges overlap widely (shadows fade out up to luma 0.6, highlights start from 0.2), so the correction changes progressively.
- **Clip Black**, **Clip White**, **Softness** (gamma of the semi-transparent values).
- **Shrink / Grow** (pixels), **Matte Blur** (pixels).
- **Reset Matte**: resets these parameters; the colours and Mix are kept.

**Patches** (× 3)
- **Patch n**: turns the patch on (done automatically by the picks).
- **Type**: Auto (chosen at the first pick: Black where the matte is mostly transparent, White where it is mostly opaque), Black, White or Edge Analysis.
- **Colour Low** / **Colour High**: range of colours of the patch (lowest and highest r, g, b).
  - **Pick Colour** replaces it with a sample (click for one pixel, drag for an area: the mean ± 1.5 standard deviations of the area).
  - **Add Colour** adds a new sample to it (Flame: resample adds, Ctrl+Alt replaces).
- **Range** (colour distance around these colours fully affected) and **Softness** (fade).
- **Reset**: patch off, all its parameters back to their defaults.

**Degrain**
- **Pick Grain**: press, then drag over a flat, grainy area of the plate. The grain size is measured, *Size* is set and Degrain is turned on.
- **Degrain**: on / off. **Size**: size of the grain. **Edges**: also smooths the luminance (cleaner but softer edges). **More**: doubles the size.
- **Reset Degrain**: Degrain off, parameters back to their defaults.

**Spill**
- **Auto CC**: automatic colour suppression (despill), on / off.
- **Strength**: suppression at the edges of the subject. **Range**: distance (pixels) over which it fades to **Interior** (amount kept deep inside the subject).
- **Hue**: moves the suppression towards a neighbouring hue (green screen: -1 warmer, +1 cooler).
- **Spill Colour** (with *Pick Colour*, picked on the Source): colour of the spill to remove. It follows the Primary until it is changed; a different colour changes the suppression, not the key.
- **Reset Spill**: resets these parameters; the Spill Colour goes back to the Primary.

**Blend**
- **Edge Size**: width (pixels) of the edge band; 0 = off.
- **Edge Luma**: darkens or brightens the band. **Edge Balance**: colour shift (only its chroma is used).
- **From BG**: pushes the band towards the colour of the blurred BG; **BG Blur**: blur of the BG.
- **Reset** (next to Edge Balance) resets the Edge Balance only; **Reset Blend** resets the other blend parameters.

### How to use

1. Connect the plate to **Source** and, if you have it, the background to **BG** (and, optionally, a denoised or graded plate to **Key In**). Set *Output* to **Alpha**.
2. Press **Pick Colour** next to *Primary* and drag over a typical part of the screen in the viewer (the average of the area is taken).
3. If the screen is uneven, either set *Screen* to **Local**, or pick the **Secondary** on the darkest part of the screen and raise **Mix** until the background is black.
4. Choose the **Matte Mode**: *Difference* for fine edges, *3D* to isolate colours precisely, *Combined* for both (a good start when light colours of the subject turn grey). In 3D, adjust *Tolerance* until the screen is black and *3D Softness* for the edges.
5. Make the subject solid: if light colours that contain some green (blond hair, yellow, white) are grey in the matte, move **Balance** towards the channel that is high in them (towards -1, red, for warm colours); or lower **Highlights**; then use **Clip White**. Clean the background with **Clip Black**, **Gain** or **Shadows**.
6. Fix what remains with **Patches**: press *Pick Colour* and drag over a grey area of the matte (the type is chosen automatically), then *Add Colour* for more colours; Edge Analysis for fringes. Use them sparingly: they can make the matte crunchy. **Garbage** / **Holdout** inputs and **Shrink / Grow** complete them.
7. On a grainy plate, press **Pick Grain** and drag over a flat area, then compare the matte edges (adjust *Size*, *Edges*, *More*).
8. Set *Output* to **Result** (or **Source CC** to see the despilled plate alone). Adjust **Spill** (lower *Interior* to keep colours close to the screen inside the subject; pick a *Spill Colour* when the spill does not have the hue of the screen).
9. Integrate the edges with **Blend**: *Edge Size* a few pixels, then *From BG*, *Edge Luma* or *Edge Balance*.

Tests on a synthetic plate (uneven green screen, skin and blond hair, 8-pixel soft edge) give these mean matte errors against the true alpha:

| Settings | Screen | Edges | Skin | Blond hair |
|---|---|---|---|---|
| Defaults (Primary only) | 0.091 | 0.138 | 0.000 | 0.221 |
| Screen = Local | 0.000 | 0.129 | 0.000 | 0.223 |
| Screen = Local, Balance -0.5 | 0.000 | 0.071 | 0.000 | 0.004 |
| Same with grain added | 0.011 | 0.083 | 0.000 | 0.011 |
| Same with grain added and Degrain | 0.000 | 0.075 | 0.000 | 0.005 |
| Matte Mode 3D, Screen = Local | 0.000 | 0.084 | 0.000 | 0.000 |
| Matte Mode Combined, Screen = Local | 0.000 | 0.073 | 0.000 | 0.000 |

Like every keyer that measures the screen colour channel by channel, MasterKeyer_OFX makes light colours that contain the screen colour (blond hair against green) partly transparent with neutral settings: *Balance* (or the *Combined* / *3D* matte modes, which keep them solid) is the first setting to adjust for them.

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
| MasterKeyer default | 70 ms | 25.6 ms | ×2.7 |
| MasterKeyer Local screen + Degrain + Shrink + Matte Blur + Blend + BG | 296 ms | 41.9 ms | ×7.1 |
| MasterKeyer Combined matte + Key In + tone corrections + 3 patches + Spill Colour | 96 ms | 33.8 ms | ×2.8 |

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
| MasterKeyer default | 241 ms | 92.3 ms | ×2.6 |
| MasterKeyer Local screen + Degrain + Shrink + Matte Blur + Blend + BG | 1370 ms | 150.9 ms | ×9.1 |
| MasterKeyer Combined matte + Key In + tone corrections + 3 patches + Spill Colour | 344 ms | 125.2 ms | ×2.7 |

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
| MasterKeyer default | 65 ms | 14.1 ms | ×4.6 |
| MasterKeyer Local screen + Degrain + Shrink + Matte Blur + Blend + BG | 279 ms | 21.8 ms | ×12.8 |
| MasterKeyer Combined matte + Key In + tone corrections + 3 patches + Spill Colour | 73 ms | 15.3 ms | ×4.8 |

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
| MasterKeyer default | 190 ms | 39.5 ms | ×4.8 |
| MasterKeyer Local screen + Degrain + Shrink + Matte Blur + Blend + BG | 720 ms | 75.1 ms | ×9.6 |
| MasterKeyer Combined matte + Key In + tone corrections + 3 patches + Spill Colour | 209 ms | 51.2 ms | ×4.1 |

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
- The OpenFX SDK (`OFX_Release_1.5.1`) is downloaded automatically from GitHub during the first configuration, which is why Git is required.
- Visual Studio 2026 needs CUDA Toolkit 13.x. The Nsight warnings shown by the CUDA installer about Visual Studio integration can be ignored.
- **Nuke versions**: tested with Nuke 14 and Nuke 17 on Windows; one build works for all versions. Nuke 14 and older ship their own, older Microsoft C++ runtime (`msvcp140.dll`), which plug-ins built with Visual Studio 2022 17.10 or newer would crash with. The CMake projects define `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` on Windows so that the plug-ins work with every Nuke version.
- The CUDA runtime is linked statically: the plug-ins only need a recent NVIDIA driver on the machine that runs Nuke, no CUDA DLL to ship.
- On Linux, the C++ runtime is linked statically too: the plug-ins only depend on glibc 2.38 or newer and, for push_IT_OFX, apGlow_OFX, despill_IT_OFX and MasterKeyer_OFX, on `libGL.so.1` (viewer overlays), which every machine running Nuke has.
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
| `PUSHIT_WITH_CUDA` / `APGLOW_WITH_CUDA` / `LDESPILL_WITH_CUDA` / `DESPILLIT_WITH_CUDA` / `MASTERKEYER_WITH_CUDA` | `ON` | Build the CUDA back-end (if a CUDA compiler is found). |
| `PUSHIT_CUDA_ARCHS` / `APGLOW_CUDA_ARCHS` / `LDESPILL_CUDA_ARCHS` / `DESPILLIT_CUDA_ARCHS` / `MASTERKEYER_CUDA_ARCHS` | empty (automatic) | GPU architectures, e.g. `"75-real;86-real"`. |
| `APGLOW_AVX2` | `OFF` | AVX2 / FMA on the CPU, about 15 % faster. The plug-in will **not load** on a CPU without AVX2 (before Intel Haswell / AMD Ryzen). |
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

The nodes appear in the **Filter** menu (push_IT_OFX, apGlow_OFX) and the **Keyer** menu (L_Despill_OFX, despill_IT_OFX, MasterKeyer_OFX), or with Tab.

**After an installation or an update, clear Nuke's OFX cache** (close Nuke first). Otherwise Nuke may keep showing the old version or not show the node at all:
- Windows: delete `%LOCALAPPDATA%\Temp\nuke\ofxplugincache`
- Linux / macOS: delete the `ofxplugincache` folder in Nuke's temporary directory (`NUKE_TEMP_DIR`, `/var/tmp/nuke` by default on Linux)

**Node colour**: OFX plug-ins cannot set their own colour in the Node Graph. Add a rule in *Preferences > Panels > Node Colors* with the node name (e.g. `push_IT_OFX`).

**Channels**: Nuke only passes the rgba layer to OFX plug-ins, so the effects work on rgba. Use a Shuffle before the node to process another layer.

---

## Viewer overlays

push_IT_OFX, apGlow_OFX, despill_IT_OFX and MasterKeyer_OFX draw widgets in the Viewer while their panel is open (see each plug-in). Like the handles of Nuke's own nodes (a Transform, for instance), they cannot be dragged while the Viewer is playing: pause the playback to use them. The parameters of the panel can still be changed during playback.

---

## GPU notes

Nuke does not give GPU buffers to OFX plug-ins. With *Use GPU* on, each frame is sent to the graphics card, processed entirely there, and copied back. The copies go through page-locked memory, overlapped with the transfers and spread over several CPU cores; apGlow_OFX only transfers the RGB channels it needs. In practice the host ↔ GPU transfers, not the processing, set the GPU render time, so the gain is largest on heavy settings (large blurs, Vector Warp, many glow steps).

If CUDA is not available or a GPU render fails, the frame is rendered on the CPU and a warning is written to Nuke's log.

---

## Benchmark (optional)

`ofx_bench` renders a set of test cases in HD and UHD on the CPU and on the GPU, prints the timings per stage and checks that both renders are identical. Nuke is not needed.

- Windows: run `ofx_bench\build_and_run.bat`. The results are written to `bench_result.txt`.
- Linux: run `ofx_bench/run.sh`.

The benchmark contains a copy of the plug-in sources (`ofx_bench/src/push_it`, `ofx_bench/src/expo_glow` (apGlow_OFX), `ofx_bench/src/l_despill`, `ofx_bench/src/despill_it`, `ofx_bench/src/master_keyer`): copy the files again after changing a plug-in.

---

## Credits

- apGlow_OFX is based on **apGlow v1.2** by Adrian Pueyo, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022.
- push_IT_OFX is inspired by the *Pixel Spread* effect of Autodesk Flame.
- L_Despill_OFX is based on the **L_Despill v05** gizmo by Justin Johnson (Luma Pictures) and the **DespillMadness** gizmo.
- MasterKeyer_OFX is inspired by the *Master Keyer* of Autodesk Flame (independent implementation; Flame's algorithm is not public).
- Built on the [OpenFX](https://github.com/AcademySoftwareFoundation/openfx) SDK.
