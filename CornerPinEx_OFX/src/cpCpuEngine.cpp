/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of CornerPinEx_OFX. GPL v2 or later, see LICENSE.
 * ***** END LICENSE BLOCK ***** */

#include "cpCpuEngine.h"

#include <algorithm>

namespace cpg {

CpuRenderer::CpuRenderer(const RenderJob& job)
    : _job(job)
    , _hasSrc(false)
{
    _pyr.nLevels = 0;
    _pyr.x1 = job.src.x1;
    _pyr.y1 = job.src.y1;
    for (int l = 0; l < CPG_MAX_LEVELS; ++l) {
        _pyr.level[l].data = NULL;
        _pyr.level[l].width = _pyr.level[l].height = 0;
        _pyr.level[l].stride = 0;
    }
}

template <class PIX>
static void
toFloat(const HostImage& im, int NC, std::vector<float>& out)
{
    const int w = im.width(), h = im.height();
    out.resize( (size_t)w * h * NC );
    for (int y = 0; y < h; ++y) {
        const PIX* row = (const PIX*)( (const unsigned char*)im.data + (long long)y * im.rowBytes );
        float* o = &out[(size_t)y * w * NC];
        for (int k = 0; k < w * NC; ++k) {
            o[k] = (float)row[k];
        }
    }
}

void
CpuRenderer::prepare()
{
    const RenderJob& job = _job;
    const int NC = job.nComp;
    _hasSrc = job.src.data && job.src.width() > 0 && job.src.height() > 0;
    if (_hasSrc) {
        const int sw = job.src.width(), sh = job.src.height();
        FImage& l0 = _pyr.level[0];
        l0.width = sw;
        l0.height = sh;
        if ( (job.depth == eDepthFloat) && (job.src.rowBytes % (long long)sizeof(float) == 0) ) {
            // use the host image directly
            l0.data = (const float*)job.src.data;
            l0.stride = job.src.rowBytes / (long long)sizeof(float);
        } else {
            _levels.push_back( std::vector<float>() );
            if (job.depth == eDepthUByte) {
                toFloat<unsigned char>(job.src, NC, _levels.back());
            } else if (job.depth == eDepthUShort) {
                toFloat<unsigned short>(job.src, NC, _levels.back());
            } else {
                toFloat<float>(job.src, NC, _levels.back());
            }
            l0.data = &_levels.back()[0];
            l0.stride = (long long)sw * NC;
        }
        _pyr.nLevels = 1;
        if ( isEWA(job.sp.filter) ) {
            _levels.reserve(CPG_MAX_LEVELS + 1); // keep the pointers valid
            int w = sw, h = sh;
            while ( (_pyr.nLevels < CPG_MAX_LEVELS) && ( (w > 1) || (h > 1) ) ) {
                int nw = (w + 1) / 2, nh = (h + 1) / 2;
                _levels.push_back( std::vector<float>( (size_t)nw * nh * NC ) );
                float* dst = &_levels.back()[0];
                const FImage prev = _pyr.level[_pyr.nLevels - 1];
                for (int j = 0; j < nh; ++j) {
                    for (int i = 0; i < nw; ++i) {
                        switch (NC) {
                        case 1: downsamplePixel<1>(prev, dst, (long long)nw * NC, i, j); break;
                        case 2: downsamplePixel<2>(prev, dst, (long long)nw * NC, i, j); break;
                        case 3: downsamplePixel<3>(prev, dst, (long long)nw * NC, i, j); break;
                        default: downsamplePixel<4>(prev, dst, (long long)nw * NC, i, j); break;
                        }
                    }
                }
                FImage& cur = _pyr.level[_pyr.nLevels];
                cur.data = dst;
                cur.width = nw;
                cur.height = nh;
                cur.stride = (long long)nw * NC;
                ++_pyr.nLevels;
                w = nw;
                h = nh;
            }
        }
    }
    const int count = job.count();
    _H.resize( std::max(count, 1) );
    for (int t = 0; t < count; ++t) {
        for (int k = 0; k < 9; ++k) {
            _H[t].m[k] = (float)job.H[t * 9 + k];
        }
    }
    _alpha.resize( job.alpha.size() );
    for (size_t i = 0; i < job.alpha.size(); ++i) {
        _alpha[i] = (float)job.alpha[i];
    }
}

template <class PIX>
static inline PIX
fromFloat(float v, float maxValue)
{
    if (maxValue == 1.f) {
        return (PIX)v;
    }
    v = v < 0.f ? 0.f : (v > maxValue ? maxValue : v);
    return (PIX)(v + 0.5f);
}

template <class PIX>
static inline float
maskValue(const HostImage& m, int x, int y, float maxValue, bool invert)
{
    if ( !m.data || (x < m.x1) || (x >= m.x2) || (y < m.y1) || (y >= m.y2) ) {
        return invert ? 1.f : 0.f;
    }
    const PIX* row = (const PIX*)( (const unsigned char*)m.data + (long long)(y - m.y1) * m.rowBytes );
    float v = (float)row[x - m.x1] / maxValue;
    return invert ? 1.f - v : v;
}

template <int NC>
void
CpuRenderer::renderRowsT(int y1, int y2) const
{
    const RenderJob& job = _job;
    const float maxValue = job.maxValue();
    BlurParams bp;
    bp.H = &_H[0];
    bp.alpha = _alpha.empty() ? NULL : &_alpha[0];
    bp.count = job.count();
    bp.motionblur = (float)job.motionblur;
    bp.maxValue = maxValue;
    const size_t bpc = job.depth == eDepthUByte ? 1 : (job.depth == eDepthUShort ? 2 : 4);
    const FImage& l0 = _pyr.level[0];
    float tmp[NC], res[NC];

    for (int y = y1; y < y2; ++y) {
        unsigned char* dstRow = (unsigned char*)job.dstData + (long long)(y - job.dstY1) * job.dstRowBytes;
        for (int x = job.rw[0]; x < job.rw[2]; ++x) {
            if (_hasSrc) {
                renderPixel<NC>(_pyr, x, y, job.sp, bp, tmp);
            } else {
                for (int c = 0; c < NC; ++c) {
                    tmp[c] = 0.f;
                }
            }
            float a = 1.f;
            if (job.masked) {
                float m = 1.f;
                if (job.doMask) {
                    switch (job.depth) {
                    case eDepthUByte: m = maskValue<unsigned char>(job.mask, x, y, maxValue, job.maskInvert); break;
                    case eDepthUShort: m = maskValue<unsigned short>(job.mask, x, y, maxValue, job.maskInvert); break;
                    default: m = maskValue<float>(job.mask, x, y, maxValue, job.maskInvert); break;
                    }
                }
                a = m * (float)job.mix;
            }
            const float* srcPix = NULL;
            if (a != 1.f) {
                int si = x - _pyr.x1, sj = y - _pyr.y1;
                if ( _hasSrc && (si >= 0) && (sj >= 0) && (si < l0.width) && (sj < l0.height) ) {
                    srcPix = l0.data + (long long)sj * l0.stride + (long long)si * NC;
                }
            }
            maskMix<NC>(tmp, srcPix, a, res);
            unsigned char* d = dstRow + (long long)(x - job.dstX1) * (long long)(bpc * NC);
            for (int c = 0; c < NC; ++c) {
                switch (job.depth) {
                case eDepthUByte: ( (unsigned char*)d )[c] = fromFloat<unsigned char>(res[c], maxValue); break;
                case eDepthUShort: ( (unsigned short*)d )[c] = fromFloat<unsigned short>(res[c], maxValue); break;
                default: ( (float*)d )[c] = res[c]; break;
                }
            }
        }
    }
}

void
CpuRenderer::renderRows(int y1, int y2) const
{
    switch (_job.nComp) {
    case 1: renderRowsT<1>(y1, y2); break;
    case 2: renderRowsT<2>(y1, y2); break;
    case 3: renderRowsT<3>(y1, y2); break;
    default: renderRowsT<4>(y1, y2); break;
    }
}
} // namespace cpg
