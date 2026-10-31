/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX, a GPU-accelerated derivative of the
 * openfx-misc CornerPin plugin <https://github.com/NatronGitHub/openfx-misc>.
 * (C) 2026 CornerPinEx_OFX contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * Render engines of CornerPinEx_OFX: a description of one render ("job") that
 * does not depend on the OFX headers, so that it can be passed to the CUDA
 * code (compiled by nvcc) and to the CPU code.
 */

#ifndef CORNERPINEX_OFX_ENGINE_H
#define CORNERPINEX_OFX_ENGINE_H

#include <string>
#include <vector>

#include "cpSampler.h"

namespace cpg {

enum PixelDepth
{
    eDepthUByte = 0,
    eDepthUShort = 1,
    eDepthFloat = 2
};

// An image as given by the OFX host: data points to pixel (x1,y1), rows may be padded,
// rowBytes may be negative.
struct HostImage
{
    const void* data;
    long long rowBytes;
    int x1, y1, x2, y2;     // bounds, in pixels
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

struct RenderJob
{
    int nComp;              // 1..4
    PixelDepth depth;
    HostImage src;          // src.data may be NULL (no input)
    HostImage mask;         // mask.data may be NULL (no masking); always 1 component
    void* dstData;          // output image: pointer to pixel (dstX1, dstY1)
    long long dstRowBytes;
    int dstX1, dstY1;
    int rw[4];              // render window x1, y1, x2, y2

    std::vector<double> H;  // count * 9 doubles: inverse transforms, pixel coords, row-major
    std::vector<double> alpha; // empty, or count weights (directional blur)
    SampleParams sp;
    double motionblur;      // 0 = no motion blur
    bool masked;            // masked plugin variant (uses mix + mask)
    bool doMask;            // a mask image is connected and applied
    bool maskInvert;
    double mix;

    int count() const { return (int)(H.size() / 9); }
    float maxValue() const { return depth == eDepthUByte ? 255.f : (depth == eDepthUShort ? 65535.f : 1.f); }
};

// GPU (CUDA). Return false and fill *err if the GPU could not be used; the caller
// then falls back to the CPU engine. Implemented in cpCuda.cu.
struct CudaDeviceInfo
{
    bool available;
    int device;
    std::string name;
    std::string message; // why CUDA is not available, or a description of the device
    size_t totalMem;
};

CudaDeviceInfo cudaQueryDevice(int requestedDevice);
bool cudaRender(const RenderJob& job, int device, std::string* err);
bool cudaCompiledIn();

// Log a CPU render time when CORNERPINEX_OFX_PROFILE=1 (same log file as the GPU renders).
void profileCpuRender(const char* what, int w, int h, double ms);
bool profilingEnabled();

} // namespace cpg

#endif // CORNERPINEX_OFX_ENGINE_H
