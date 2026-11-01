// ChromaAberrations_OFX - CUDA renderer stub for builds without the CUDA Toolkit
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "render_gpu.h"

namespace ca {

struct GpuRenderer::Impl {};

GpuRenderer::GpuRenderer() : _d(0) {}
GpuRenderer::~GpuRenderer() {}

bool GpuRenderer::available(std::string* name)
{
    if (name) {
        *name = "built without CUDA";
    }
    return false;
}

bool GpuRenderer::setPlan(const LensPlan&, std::string* error)
{
    if (error) {
        *error = "built without CUDA";
    }
    return false;
}

bool GpuRenderer::render(const ImageRGB&, ImageRGB&, RenderStats*, std::string* error)
{
    if (error) {
        *error = "built without CUDA";
    }
    return false;
}

float* GpuRenderer::inputBuffer(int, int) { return 0; }
const float* GpuRenderer::outputBuffer() const { return 0; }

bool GpuRenderer::renderBuffers(int, int, RenderStats*, std::string* error)
{
    if (error) {
        *error = "built without CUDA";
    }
    return false;
}

void GpuRenderer::setKernelCacheBudget(size_t) {}
void GpuRenderer::setWavelengthBatch(int) {}

} // namespace ca
