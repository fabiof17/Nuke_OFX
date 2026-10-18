# Nuke OFX Plug-ins

A collection of OpenFX plug-ins for The Foundry Nuke, written in C++ with optional NVIDIA CUDA acceleration.

| Plug-in | What it does |
|---|---|
| [**push_IT_OFX**](#push_it_ofx) | Edge extend / pixel spread: pushes the colours of a matte outward, or fills a matted area from its surroundings. |
| [**expo_GLOW_OFX**](#expo_glow_ofx) | Exponential glow: a sum of blurs whose sizes grow geometrically, for a soft and wide light falloff. |

The plug-ins run on Windows, Linux and macOS. On Windows and Linux they can render on an NVIDIA GPU with CUDA; when no GPU is available they fall back to a multithreaded CPU renderer automatically. The CPU and GPU renders are identical.

Ready-to-use builds for Windows and Linux are available on the [Releases](../../releases) page.

Each plug-in lives in its own folder and is built on its own:

```
push_IT_OFX/      edge extend plug-in (sources, CMakeLists.txt, build.bat, build.sh)
expo_GLOW_OFX/    glow plug-in (sources, CMakeLists.txt, build.bat, build.sh)
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

## Performance

Render time per frame, CPU vs GPU, measured with `ofx_bench` (best of 5 runs). GPU times include the host ↔ GPU transfers, as in Nuke. CPU and GPU renders are identical in every test.

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
| push_IT Interpolate | 72 ms | 21.5 ms | ×3.4 |
| push_IT Stretch | 80 ms | 27.6 ms | ×2.9 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 136 ms | 29.2 ms | ×4.7 |
| push_IT Vector Warp 300 px | 149 ms | 25.9 ms | ×5.8 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 129 ms | 28.0 ms | ×4.6 |
| push_IT Interpolate + Spread Blur 500 | 148 ms | 28.4 ms | ×5.2 |
| expo_GLOW Max Size 2000, gaussian, 10 steps | 156 ms | 27.5 ms | ×5.7 |
| expo_GLOW Max Size 2000, box, 10 steps | 104 ms | 24.3 ms | ×4.3 |
| expo_GLOW Max Size 300, gaussian, 20 steps | 376 ms | 33.7 ms | ×11.2 |

**UHD (3840×2160)**

| Test | CPU | GPU | GPU speed-up |
|---|---|---|---|
| push_IT Interpolate | 250 ms | 82.3 ms | ×3.0 |
| push_IT Stretch | 251 ms | 89.3 ms | ×2.8 |
| push_IT Stretch + Max Distance + Spread Blur 50 | 496 ms | 103.6 ms | ×4.8 |
| push_IT Vector Warp 300 px | 490 ms | 93.0 ms | ×5.3 |
| push_IT Contract (Edge Shrink 5, Edge Softness 5) | 338 ms | 94.7 ms | ×3.6 |
| push_IT Interpolate + Spread Blur 500 | 491 ms | 98.2 ms | ×5.0 |
| expo_GLOW Max Size 2000, gaussian, 10 steps | 732 ms | 99.0 ms | ×7.4 |
| expo_GLOW Max Size 2000, box, 10 steps | 389 ms | 96.9 ms | ×4.0 |
| expo_GLOW Max Size 300, gaussian, 20 steps | 2024 ms | 124.6 ms | ×16.2 |

The GPU is 3 to 16 times faster than the CPU. The gain grows with the cost of the settings (large blurs, Vector Warp, many glow steps), because most of the GPU time is spent moving the image to and from the graphics card: 70 to 80 % in these tests. This test machine is a 2010-era workstation with PCI Express 2.0: on a recent PC (PCI Express 4.0 or newer, DDR4 / DDR5) the transfers are much faster, so the GPU times should be lower and the speed-ups higher.

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
| `PUSHIT_WITH_CUDA` / `EXPOGLOW_WITH_CUDA` | `ON` | Build the CUDA back-end (if a CUDA compiler is found). |
| `PUSHIT_CUDA_ARCHS` / `EXPOGLOW_CUDA_ARCHS` | empty (automatic) | GPU architectures, e.g. `"75-real;86-real"`. |
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

The nodes appear in the **Filter** menu (e.g. **Filter > push_IT_OFX**), or with Tab.

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

The benchmark contains a copy of the plug-in sources (`ofx_bench/src/push_it`, `ofx_bench/src/expo_glow`): copy the files again after changing a plug-in.

---

## Credits

- expo_GLOW_OFX is based on **apGlow v1.2** by Adrian Pueyo, [adrianpueyo.com](https://www.adrianpueyo.com), 2015-2022.
- push_IT_OFX is inspired by the *Pixel Spread* effect of Autodesk Flame.
- Built on the [OpenFX](https://github.com/AcademySoftwareFoundation/openfx) SDK.
