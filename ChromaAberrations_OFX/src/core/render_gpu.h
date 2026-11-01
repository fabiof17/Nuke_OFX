// ChromaAberrations_OFX - CUDA renderer (same pipeline as renderCpu, with cuFFT)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#pragma once

#include <string>

#include "render.h"

namespace ca {

class GpuRenderer
{
public:
    GpuRenderer();
    ~GpuRenderer();

    // false when the build has no CUDA or no usable device; 'name' receives the GPU name
    // or the reason
    static bool available(std::string* name = 0);

    // Uploads the lens plan (kernels, coefficients, maps). Call again when it changes.
    bool setPlan(const LensPlan& plan, std::string* error = 0);

    // Renders one image with the current plan. The spectra of the basis PSFs are kept on the
    // GPU between frames when they fit in the memory budget (then each frame only transforms
    // the image).
    bool render(const ImageRGB& in, ImageRGB& out, RenderStats* stats = 0, std::string* error = 0);

    // Same, with the image in page-locked host buffers owned by the renderer (fastest
    // transfers): fill the 3 planes of inputBuffer(W, H), call renderBuffers, read the
    // 3 planes of outputBuffer(). Planes are W x H floats, rows bottom-up.
    float* inputBuffer(int width, int height);
    const float* outputBuffer() const;
    bool renderBuffers(int width, int height, RenderStats* stats = 0, std::string* error = 0);

    // GPU memory the kernel spectra may use (bytes); 0 = never keep them
    void setKernelCacheBudget(size_t bytes);
    // Wavelengths per batch (0 = as many as fit in half of the free GPU memory). For tests.
    void setWavelengthBatch(int n);

private:
    bool renderPlanes(int width, int height, const float* const in[3], float* const out[3], RenderStats* stats, std::string* error);
    GpuRenderer(const GpuRenderer&);
    GpuRenderer& operator=(const GpuRenderer&);
    struct Impl;
    Impl* _d;
};

} // namespace ca
