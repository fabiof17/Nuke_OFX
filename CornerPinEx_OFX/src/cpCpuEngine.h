/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX. GPL v2 or later, see LICENSE.
 * ***** END LICENSE BLOCK ***** */

/*
 * CPU render engine of CornerPinEx_OFX, using the same sampling code as the GPU
 * (cpSampler.h). It is used for the HQ (EWA) filters when the GPU is not
 * available, and as a reference in the tests. Classic filters without GPU are
 * rendered by the original Natron code (Transform3x3Processor).
 *
 * Usage: construct, call prepare() once, then renderRows() on disjoint row
 * ranges, possibly from several threads at the same time.
 */

#ifndef CORNERPINEX_OFX_CPUENGINE_H
#define CORNERPINEX_OFX_CPUENGINE_H

#include <vector>

#include "cpEngine.h"

namespace cpg {

class CpuRenderer
{
public:
    explicit CpuRenderer(const RenderJob& job);

    void prepare();

    // render rows [y1, y2) of the render window (absolute pixel coordinates)
    void renderRows(int y1, int y2) const;

private:
    template <int NC>
    void renderRowsT(int y1, int y2) const;

    const RenderJob& _job;
    Pyramid _pyr;
    bool _hasSrc;
    std::vector<std::vector<float> > _levels; // owned float levels
    std::vector<Mat3> _H;
    std::vector<float> _alpha;
};
} // namespace cpg

#endif
