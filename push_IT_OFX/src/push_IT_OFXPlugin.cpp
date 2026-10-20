// push_IT_OFXPlugin.cpp
// OFX front-end of push_IT_OFX: an edge extend / pixel push effect for The Foundry Nuke.
//
// Inputs : Source, Matte (optional), Vector (optional)
// Types  : Interpolate (push-pull fill), Stretch (nearest edge pixel),
//          Vector Warp (directional spread along a vector input or a fixed angle)
// Modes  : Expand (push colours out of the matte) / Contract (fill the matte from outside)
#include "ofxsImageEffect.h"
#include "OverlayGL.h"
#include "ofxsMultiThread.h"
#include "ofxsLog.h"

#include "push_IT_OFXCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define kPluginName          "push_IT_OFX"
#define kPluginGrouping      "Filter"
#define kPluginIdentifier    "fr.pushit.push_IT_OFX"
#define kPluginVersionMajor  1
#define kPluginVersionMinor  0

#define kPluginDescription \
    "push_IT_OFX: edge extend / pixel spread.\n\n" \
    "Expand: pushes the matte colours outwards (edge extend, edge despill).\n" \
    "Contract: fills the matte area with the surrounding pixels (quick clean plate).\n\n" \
    "Types: Interpolate (smooth push-pull fill), Stretch (stretches the nearest edge pixel), " \
    "Vector Warp (directional stretch along a vector or an angle).\n\n" \
    "Optional CUDA GPU acceleration (NVIDIA cards), automatic fallback to the CPU."

// clips
static const char* kClipMatte  = "Matte";
static const char* kClipVector = "Vector";

// params
static const char* kParamSpreadType      = "spreadType";
static const char* kParamSpreadMode      = "spreadMode";
static const char* kParamMatteSource     = "matteSource";
static const char* kParamInvertMatte     = "invertMatte";
static const char* kParamSourcePremult   = "sourcePremultiplied";
static const char* kParamShrink          = "edgeShrink";
static const char* kParamEdgeSoftness    = "edgeSoftness";
static const char* kParamSeedThreshold   = "seedThreshold";
static const char* kParamMaxDistance     = "maxDistance";
static const char* kParamFalloff         = "distanceFalloff";
static const char* kParamBlur            = "spreadBlur";
static const char* kParamVectorSource    = "vectorSource";
static const char* kParamAngle           = "spreadAngle";
static const char* kParamLength          = "spreadLength";
static const char* kParamMotionDistance  = "motionDistance";
static const char* kParamMotionThreshold = "motionThreshold";
static const char* kParamVectorOrigin    = "vectorOrigin";
static const char* kParamGainRX = "redXGain";
static const char* kParamGainRY = "redYGain";
static const char* kParamGainGX = "greenXGain";
static const char* kParamGainGY = "greenYGain";
static const char* kParamGainBX = "blueXGain";
static const char* kParamGainBY = "blueYGain";
static const char* kParamInvertDirection = "invertDirection";
static const char* kParamResultOutput    = "resultOutput";
static const char* kParamAlphaOutput     = "alphaOutput";
static const char* kParamPremultOutput   = "premultiplyOutput";
static const char* kParamExtendToFormat  = "extendToFormat";
static const char* kParamUseGPU          = "useGPU";
static const char* kParamGPUStatus       = "gpuStatus";
static const char* kParamShowOverlay     = "showOverlay";
static const char* kParamOverlayOffset   = "overlayOffset";

// Layout hint (host extension, not in the base OFX headers).
static const char* kPropLayoutHint = "OfxParamPropLayoutHint";
static const int   kLayoutHintNoNewLine = 2;
static const int   kLayoutHintDivider   = 1;   // horizontal line after the parameter

enum MatteSourceChoice { kMatteAuto = 0, kMatteSourceAlpha, kMatteMatteAlpha, kMatteMatteLuminance };

using namespace pushit;

// ============================================================ image helpers
namespace {

// Reads one pixel of an OFX float image into rgba (missing channels: 0, alpha 1 for RGB).
inline void readPixel(const float* px, OFX::PixelComponentEnum comps, float* rgba)
{
    switch (comps) {
    case OFX::ePixelComponentRGBA:  rgba[0] = px[0]; rgba[1] = px[1]; rgba[2] = px[2]; rgba[3] = px[3]; break;
    case OFX::ePixelComponentRGB:   rgba[0] = px[0]; rgba[1] = px[1]; rgba[2] = px[2]; rgba[3] = 1.f;   break;
    case OFX::ePixelComponentAlpha: rgba[0] = rgba[1] = rgba[2] = 0.f; rgba[3] = px[0];               break;
    default:                        rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.f;                      break;
    }
}

inline int choiceAt(OFX::ChoiceParam* p, double t)
{
    int v = 0;
    p->getValueAtTime(t, v);
    return v;
}

inline int choiceNow(OFX::ChoiceParam* p)
{
    int v = 0;
    p->getValue(v);
    return v;
}

inline int componentCount(OFX::PixelComponentEnum c)
{
    switch (c) {
    case OFX::ePixelComponentRGBA:  return 4;
    case OFX::ePixelComponentRGB:   return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default:                        return 0;
    }
}

// Runs f(y0, y1) on contiguous row ranges of [0, n) across hardware threads.
template <class F>
void parallelRanges(int n, F&& f)
{
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 64 || nt == 1) { f(0, n); return; }
    nt = std::min<unsigned>(nt, (unsigned)n);
    const int chunk = (n + (int)nt - 1) / (int)nt;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t) {
        const int b = (int)t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    for (auto& th : pool) th.join();
}

// Rows [r0, r1) of `win` (0 = bottom row of win) of `img` as dense RGBA, zeros outside the
// image bounds. Thread-safe: only reads the image.
void readRowsRGBA(const OFX::Image* img, const OfxRectI& win, int r0, int r1, float* out)
{
    const int w = win.x2 - win.x1;
    const size_t rowFloats = 4 * (size_t)w;
    const OFX::PixelComponentEnum comps = img ? img->getPixelComponents() : OFX::ePixelComponentNone;
    const int nc = img ? componentCount(comps) : 0;
    const OfxRectI bd = img ? img->getBounds() : OfxRectI{ 0, 0, 0, 0 };
    const int x1 = std::max(win.x1, bd.x1), x2 = std::min(win.x2, bd.x2);
    for (int r = r0; r < r1; ++r) {
        float* rowOut = out + (size_t)(r - r0) * rowFloats;
        const int y = win.y1 + r;
        const float* px = nc && x1 < x2 && y >= bd.y1 && y < bd.y2
                              ? static_cast<const float*>(img->getPixelAddress(x1, y)) : nullptr;
        // zeros only where the image does not cover the row (no full clear of the buffer)
        if (!px) { std::memset(rowOut, 0, sizeof(float) * rowFloats); continue; }
        if (x1 > win.x1) std::memset(rowOut, 0, sizeof(float) * 4 * (size_t)(x1 - win.x1));
        if (x2 < win.x2) std::memset(rowOut + 4 * (size_t)(x2 - win.x1), 0, sizeof(float) * 4 * (size_t)(win.x2 - x2));
        float* o = rowOut + 4 * (size_t)(x1 - win.x1);
        if (nc == 4) { std::memcpy(o, px, sizeof(float) * 4 * (size_t)(x2 - x1)); continue; }
        for (int x = x1; x < x2; ++x, px += nc, o += 4) readPixel(px, comps, o);
    }
}

// Same, one channel: alpha, or luminance (luma && the image has colour).
void readRowsMatte(const OFX::Image* img, const OfxRectI& win, int r0, int r1, bool luma, float* out)
{
    const int w = win.x2 - win.x1;
    std::memset(out, 0, sizeof(float) * (size_t)w * (r1 - r0));
    if (!img) return;
    const OFX::PixelComponentEnum comps = img->getPixelComponents();
    const int nc = componentCount(comps);
    if (nc == 0) return;
    const bool useLuma = luma && comps != OFX::ePixelComponentAlpha;
    const OfxRectI bd = img->getBounds();
    const int x1 = std::max(win.x1, bd.x1), x2 = std::min(win.x2, bd.x2);
    if (x1 >= x2) return;
    for (int r = r0; r < r1; ++r) {
        const int y = win.y1 + r;
        if (y < bd.y1 || y >= bd.y2) continue;
        const float* px = static_cast<const float*>(img->getPixelAddress(x1, y));
        if (!px) continue;
        float* o = out + (size_t)(r - r0) * w + (x1 - win.x1);
        for (int x = x1; x < x2; ++x, px += nc, ++o) {
            float m[4];
            readPixel(px, comps, m);
            *o = useLuma ? 0.2126f * m[0] + 0.7152f * m[1] + 0.0722f * m[2] : m[3];
        }
    }
}

// Dense RGBA rows [r0, r1) of `win` into the output image.
void writeRowsRGBA(OFX::Image* img, const OfxRectI& win, int r0, int r1, const float* in)
{
    const int w = win.x2 - win.x1;
    const OFX::PixelComponentEnum comps = img->getPixelComponents();
    const int nc = componentCount(comps);
    const OfxRectI bd = img->getBounds();
    const int x1 = std::max(win.x1, bd.x1), x2 = std::min(win.x2, bd.x2);
    if (x1 >= x2 || nc == 0) return;
    for (int r = r0; r < r1; ++r) {
        const int y = win.y1 + r;
        if (y < bd.y1 || y >= bd.y2) continue;
        float* px = static_cast<float*>(img->getPixelAddress(x1, y));
        if (!px) continue;
        const float* s = in + 4 * ((size_t)(r - r0) * w + (x1 - win.x1));
        if (nc == 4) { std::memcpy(px, s, sizeof(float) * 4 * (size_t)(x2 - x1)); continue; }
        for (int x = x1; x < x2; ++x, px += nc, s += 4) {
            if (nc == 3) { px[0] = s[0]; px[1] = s[1]; px[2] = s[2]; }
            else         { px[0] = s[3]; }
        }
    }
}

} // namespace

// ================================================================== plugin
class push_IT_OFXPlugin : public OFX::ImageEffect
{
public:
    explicit push_IT_OFXPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;
    void changedClip(const OFX::InstanceChangedArgs& args, const std::string& clipName) override;

private:
    void updateParamsVisibility();
    void fullFrame(double time, OfxRectD& rod);

    OFX::Clip* _dstClip = nullptr;
    OFX::Clip* _srcClip = nullptr;
    OFX::Clip* _matteClip = nullptr;
    OFX::Clip* _vectorClip = nullptr;

    OFX::ChoiceParam*   _spreadType;
    OFX::ChoiceParam*   _spreadMode;
    OFX::ChoiceParam*   _matteSource;
    OFX::BooleanParam*  _invertMatte;
    OFX::BooleanParam*  _sourcePremult;
    OFX::DoubleParam*   _shrink;
    OFX::DoubleParam*   _edgeSoftness;
    OFX::DoubleParam*   _seedThreshold;
    OFX::DoubleParam*   _maxDistance;
    OFX::DoubleParam*   _falloff;
    OFX::DoubleParam*   _blur;
    OFX::ChoiceParam*   _vectorSource;
    OFX::DoubleParam*   _angle;
    OFX::DoubleParam*   _length;
    OFX::DoubleParam*   _motionDistance;
    OFX::DoubleParam*   _motionThreshold;
    OFX::Double2DParam* _vectorOrigin;
    OFX::DoubleParam*   _gRX; OFX::DoubleParam* _gRY;
    OFX::DoubleParam*   _gGX; OFX::DoubleParam* _gGY;
    OFX::DoubleParam*   _gBX; OFX::DoubleParam* _gBY;
    OFX::BooleanParam*  _invertDirection;
    OFX::ChoiceParam*   _resultOutput;
    OFX::ChoiceParam*   _alphaOutput;
    OFX::BooleanParam*  _premultOutput;
    OFX::BooleanParam*  _extendToFormat;
    OFX::BooleanParam*  _useGPU;
    OFX::StringParam*   _gpuStatus;
};

push_IT_OFXPlugin::push_IT_OFXPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _matteClip = fetchClip(kClipMatte);
    _vectorClip = fetchClip(kClipVector);

    _spreadType      = fetchChoiceParam(kParamSpreadType);
    _spreadMode      = fetchChoiceParam(kParamSpreadMode);
    _matteSource     = fetchChoiceParam(kParamMatteSource);
    _invertMatte     = fetchBooleanParam(kParamInvertMatte);
    _sourcePremult   = fetchBooleanParam(kParamSourcePremult);
    _shrink          = fetchDoubleParam(kParamShrink);
    _edgeSoftness    = fetchDoubleParam(kParamEdgeSoftness);
    _seedThreshold   = fetchDoubleParam(kParamSeedThreshold);
    _maxDistance     = fetchDoubleParam(kParamMaxDistance);
    _falloff         = fetchDoubleParam(kParamFalloff);
    _blur            = fetchDoubleParam(kParamBlur);
    _vectorSource    = fetchChoiceParam(kParamVectorSource);
    _angle           = fetchDoubleParam(kParamAngle);
    _length          = fetchDoubleParam(kParamLength);
    _motionDistance  = fetchDoubleParam(kParamMotionDistance);
    _motionThreshold = fetchDoubleParam(kParamMotionThreshold);
    _vectorOrigin    = fetchDouble2DParam(kParamVectorOrigin);
    _gRX = fetchDoubleParam(kParamGainRX); _gRY = fetchDoubleParam(kParamGainRY);
    _gGX = fetchDoubleParam(kParamGainGX); _gGY = fetchDoubleParam(kParamGainGY);
    _gBX = fetchDoubleParam(kParamGainBX); _gBY = fetchDoubleParam(kParamGainBY);
    _invertDirection = fetchBooleanParam(kParamInvertDirection);
    _resultOutput    = fetchChoiceParam(kParamResultOutput);
    _alphaOutput     = fetchChoiceParam(kParamAlphaOutput);
    _premultOutput   = fetchBooleanParam(kParamPremultOutput);
    _extendToFormat  = fetchBooleanParam(kParamExtendToFormat);
    _useGPU          = fetchBooleanParam(kParamUseGPU);
    _gpuStatus       = fetchStringParam(kParamGPUStatus);

    updateParamsVisibility();
}

void push_IT_OFXPlugin::updateParamsVisibility()
{
    const int type = choiceNow(_spreadType);
    const bool vecWarp = type == kTypeVectorWarp;
    const bool constant = choiceNow(_vectorSource) == kVectorConstant;
    _seedThreshold->setEnabled(type == kTypeStretch || _maxDistance->getValue() > 0.0);
    _falloff->setEnabled(_maxDistance->getValue() > 0.0);
    _vectorSource->setEnabled(vecWarp);
    _angle->setEnabled(vecWarp && constant);
    _length->setEnabled(vecWarp && constant);
    _motionDistance->setEnabled(vecWarp && !constant);
    _vectorOrigin->setEnabled(vecWarp && !constant);
    OFX::DoubleParam* gains[] = { _gRX, _gRY, _gGX, _gGY, _gBX, _gBY };
    for (OFX::DoubleParam* g : gains) g->setEnabled(vecWarp && !constant);
    _motionThreshold->setEnabled(vecWarp);
    _invertDirection->setEnabled(vecWarp);
    _useGPU->setEnabled(cudaIsCompiledIn());
}

void push_IT_OFXPlugin::changedParam(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kParamSpreadType || name == kParamVectorSource || name == kParamMaxDistance)
        updateParamsVisibility();
}

void push_IT_OFXPlugin::changedClip(const OFX::InstanceChangedArgs&, const std::string&)
{
    updateParamsVisibility();
}

void push_IT_OFXPlugin::fullFrame(double time, OfxRectD& rod)
{
    rod = _srcClip && _srcClip->isConnected() ? _srcClip->getRegionOfDefinition(time) : OfxRectD{ 0, 0, 0, 0 };
    if (_extendToFormat->getValueAtTime(time)) {
        const OfxPointD off = getProjectOffset(), ext = getProjectExtent();
        const bool emptyRod = rod.x2 <= rod.x1 || rod.y2 <= rod.y1;
        rod.x1 = emptyRod ? off.x : std::min(rod.x1, off.x);
        rod.y1 = emptyRod ? off.y : std::min(rod.y1, off.y);
        rod.x2 = emptyRod ? off.x + ext.x : std::max(rod.x2, off.x + ext.x);
        rod.y2 = emptyRod ? off.y + ext.y : std::max(rod.y2, off.y + ext.y);
    }
}

bool push_IT_OFXPlugin::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if (!_extendToFormat->getValueAtTime(args.time)) return false;   // = source RoD
    fullFrame(args.time, rod);
    return true;
}

// The spread is global: every output pixel may depend on any input pixel.
void push_IT_OFXPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois)
{
    if (_srcClip->isConnected()) rois.setRegionOfInterest(*_srcClip, _srcClip->getRegionOfDefinition(args.time));
    if (_matteClip && _matteClip->isConnected()) rois.setRegionOfInterest(*_matteClip, _matteClip->getRegionOfDefinition(args.time));
    if (_vectorClip && _vectorClip->isConnected()) rois.setRegionOfInterest(*_vectorClip, _vectorClip->getRegionOfDefinition(args.time));
}

void push_IT_OFXPlugin::render(const OFX::RenderArguments& args)
{
    const double t = args.time;
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(t));
    if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (dst->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);

    // Process the whole output image (tiles are disabled, but be defensive).
    OfxRectI win = dst->getBounds();
    const int w = win.x2 - win.x1, h = win.y2 - win.y1;
    if (w <= 0 || h <= 0) return;

    std::unique_ptr<const OFX::Image> src(_srcClip->isConnected() ? _srcClip->fetchImage(t) : nullptr);
    const bool matteConnected = _matteClip && _matteClip->isConnected();
    const bool vectorConnected = _vectorClip && _vectorClip->isConnected();
    std::unique_ptr<const OFX::Image> matteImg(matteConnected ? _matteClip->fetchImage(t) : nullptr);

    if (src && src->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    if (matteImg && matteImg->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);

    // ---- matte channel (nullptr = source alpha)
    int matteChoice = choiceAt(_matteSource, t);
    if (matteChoice == kMatteAuto) matteChoice = matteConnected ? kMatteMatteAlpha : kMatteSourceAlpha;
    const bool matteFromSource = matteChoice == kMatteSourceAlpha;
    const bool matteLuma = matteChoice == kMatteMatteLuminance;

    // ---- parameters (pixel values follow the render scale, e.g. proxy mode)
    const double sx = args.renderScale.x, sy = args.renderScale.y;
    const double sc = 0.5 * (sx + sy);
    Params p;
    p.width = w; p.height = h;
    p.spreadType = choiceAt(_spreadType, t);
    p.spreadMode = choiceAt(_spreadMode, t);
    p.invertMatte = _invertMatte->getValueAtTime(t);
    p.sourcePremultiplied = _sourcePremult->getValueAtTime(t);
    p.shrink = (float)(_shrink->getValueAtTime(t) * sc);
    p.edgeSoftness = (float)(_edgeSoftness->getValueAtTime(t) * sc);
    p.seedThreshold = (float)_seedThreshold->getValueAtTime(t);
    p.maxDistance = (float)(_maxDistance->getValueAtTime(t) * sc);
    p.falloff = (float)(_falloff->getValueAtTime(t) * sc);
    p.blur = (float)(_blur->getValueAtTime(t) * sc);
    p.vectorSource = choiceAt(_vectorSource, t);
    const double ang = _angle->getValueAtTime(t) * 3.14159265358979323846 / 180.0;
    const double len = _length->getValueAtTime(t);
    p.constX = (float)(std::cos(ang) * len * sx);
    p.constY = (float)(std::sin(ang) * len * sy);
    p.motionDistance = (float)(_motionDistance->getValueAtTime(t) * sc);
    p.motionThreshold = (float)(_motionThreshold->getValueAtTime(t) * sc);
    double ox, oy;
    _vectorOrigin->getValueAtTime(t, ox, oy);
    p.originX = (float)ox; p.originY = (float)oy;
    p.gainRX = (float)_gRX->getValueAtTime(t); p.gainRY = (float)_gRY->getValueAtTime(t);
    p.gainGX = (float)_gGX->getValueAtTime(t); p.gainGY = (float)_gGY->getValueAtTime(t);
    p.gainBX = (float)_gBX->getValueAtTime(t); p.gainBY = (float)_gBY->getValueAtTime(t);
    p.invertDirection = _invertDirection->getValueAtTime(t);
    p.resultOutput = choiceAt(_resultOutput, t);
    p.alphaOutput = choiceAt(_alphaOutput, t);
    p.premultiplyOutput = _premultOutput->getValueAtTime(t);

    // ---- vector input
    std::unique_ptr<const OFX::Image> vecImg;
    if (p.spreadType == kTypeVectorWarp && p.vectorSource == kVectorInput && vectorConnected) {
        vecImg.reset(_vectorClip->fetchImage(t));
        if (vecImg && vecImg->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    }

    // ---- GPU: the images are converted straight into / out of the transfer memory
    bool done = false;
    if (_useGPU->getValueAtTime(t) && cudaIsCompiledIn()) {
        const RowReader rSrc = [&](int r0, int r1, float* o) { readRowsRGBA(src.get(), win, r0, r1, o); };
        const RowReader rMatte = [&](int r0, int r1, float* o) { readRowsMatte(matteImg.get(), win, r0, r1, matteLuma, o); };
        const RowReader rVec = [&](int r0, int r1, float* o) { readRowsRGBA(vecImg.get(), win, r0, r1, o); };
        const RowWriter wDst = [&](int r0, int r1, const float* i) { writeRowsRGBA(dst.get(), win, r0, r1, i); };
        std::string err;
        done = processCUDARows(p, rSrc, matteFromSource ? nullptr : &rMatte, vecImg ? &rVec : nullptr, wDst, err);
        if (!done) OFX::Log::warning(true, "push_IT_OFX: GPU render failed (%s), falling back to CPU", err.c_str());
    }
    if (done) return;

    // ---- CPU
    const size_t N = (size_t)w * h;
    std::vector<float> srcBuf(4 * N), matte, vecBuf, dstBuf(4 * N);
    parallelRanges(h, [&](int r0, int r1) { readRowsRGBA(src.get(), win, r0, r1, &srcBuf[4 * (size_t)r0 * w]); });
    if (!matteFromSource) {
        matte.resize(N);
        parallelRanges(h, [&](int r0, int r1) { readRowsMatte(matteImg.get(), win, r0, r1, matteLuma, &matte[(size_t)r0 * w]); });
    }
    if (vecImg) {
        vecBuf.resize(4 * N);
        parallelRanges(h, [&](int r0, int r1) { readRowsRGBA(vecImg.get(), win, r0, r1, &vecBuf[4 * (size_t)r0 * w]); });
    }
    processCPU(p, srcBuf.data(), matteFromSource ? nullptr : matte.data(), vecImg ? vecBuf.data() : nullptr, dstBuf.data());
    parallelRanges(h, [&](int r0, int r1) { writeRowsRGBA(dst.get(), win, r0, r1, &dstBuf[4 * (size_t)r0 * w]); });
}

// ================================================================= factory
// ================================================================== viewer overlay
// Drawn while the node panel is open (Show Overlay on):
//   - a centre cross, draggable, where the widgets are anchored (format centre + offset);
//   - Vector Warp with a constant vector: an arrow showing Angle and Length; drag its tip;
//   - Max Distance > 0: a dashed circle of that radius; drag its handle to change it.
namespace {

template <class P, class... V>
void setNow(P* p, double t, V... v)
{
    if (p->getNumKeys() > 0) p->setValueAtTime(t, v...);
    else p->setValue(v...);
}

class PushOverlay : public OFX::OverlayInteract
{
public:
    PushOverlay(OfxInteractHandle handle, OFX::ImageEffect* effect) : OFX::OverlayInteract(handle)
    {
        _effect = effect;
        _show = effect->fetchBooleanParam(kParamShowOverlay);
        _offset = effect->fetchDouble2DParam(kParamOverlayOffset);
        _type = effect->fetchChoiceParam(kParamSpreadType);
        _source = effect->fetchChoiceParam(kParamVectorSource);
        _angle = effect->fetchDoubleParam(kParamAngle);
        _length = effect->fetchDoubleParam(kParamLength);
        _invert = effect->fetchBooleanParam(kParamInvertDirection);
        _maxDist = effect->fetchDoubleParam(kParamMaxDistance);
        _src = effect->fetchClip(kOfxImageEffectSimpleSourceClipName);
        for (OFX::Param* q : std::initializer_list<OFX::Param*>{ _show, _offset, _type, _source, _angle, _length, _invert, _maxDist })
            addParamToSlaveTo(q);
    }

    bool draw(const OFX::DrawArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        const State st = state(args.time);
        const ovl::Pen pen(args.pixelScale);
        const ovl::GLState gl;
        const ovl::Colour base = ovl::baseColour(*this);
        auto col = [&](Part p) { return _drag == p || _hover == p ? ovl::highlight() : base; };
        if (st.maxDist > 0.0) {
            ovl::circle(pen, col(kRadius), st.cx, st.cy, st.maxDist * st.par, st.maxDist, 1.5f, true);
            ovl::handle(pen, col(kRadius), st.cx + st.maxDist * st.par, st.cy);
        }
        if (st.arrow) {
            ovl::arrow(pen, col(kTip), st.cx, st.cy, st.tx, st.ty);
            ovl::handle(pen, col(kTip), st.tx, st.ty);
        }
        ovl::cross(pen, col(kCentre), st.cx, st.cy);
        return true;
    }

    bool penMotion(const OFX::PenArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        const double x = args.penPosition.x, y = args.penPosition.y;
        const State st = state(args.time);
        if (_drag == kNone) {
            const Part h = hit(st, ovl::Pen(args.pixelScale), x, y);
            if (h != _hover) { _hover = h; requestRedraw(); }
            return false;
        }
        if (_drag == kCentre) {
            setNow(_offset, args.time, _dragOffX + (x - _dragX), _dragOffY + (y - _dragY));
        } else if (_drag == kTip) {
            double vx = (x - st.cx) / st.par, vy = y - st.cy;
            if (_invert->getValueAtTime(args.time)) { vx = -vx; vy = -vy; }
            setNow(_length, args.time, std::sqrt(vx * vx + vy * vy));
            setNow(_angle, args.time, std::atan2(vy, vx) * 180.0 / 3.14159265358979323846);
        } else if (_drag == kRadius) {
            const double dx = (x - st.cx) / st.par, dy = y - st.cy;
            setNow(_maxDist, args.time, std::sqrt(dx * dx + dy * dy));
        }
        requestRedraw();
        return true;
    }

    bool penDown(const OFX::PenArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        const State st = state(args.time);
        _drag = hit(st, ovl::Pen(args.pixelScale), args.penPosition.x, args.penPosition.y);
        if (_drag == kNone) return false;
        _dragX = args.penPosition.x; _dragY = args.penPosition.y;
        _offset->getValueAtTime(args.time, _dragOffX, _dragOffY);
        requestRedraw();
        return true;
    }

    bool penUp(const OFX::PenArgs&) override
    {
        if (_drag == kNone) return false;
        _drag = kNone;
        requestRedraw();
        return true;
    }

private:
    enum Part { kNone, kCentre, kTip, kRadius };
    struct State { double cx, cy, tx, ty, par, maxDist; bool arrow; };

    State state(double t) const
    {
        State s;
        const OfxPointD off = _effect->getProjectOffset(), ext = _effect->getProjectExtent();
        double ox, oy;
        _offset->getValueAtTime(t, ox, oy);
        s.cx = off.x + 0.5 * ext.x + ox;
        s.cy = off.y + 0.5 * ext.y + oy;
        s.par = _src->isConnected() ? _src->getPixelAspectRatio() : 1.0;
        if (!(s.par > 0.0)) s.par = 1.0;
        int type = 0, source = 0;
        _type->getValueAtTime(t, type);
        _source->getValueAtTime(t, source);
        s.arrow = type == pushit::kTypeVectorWarp && source == pushit::kVectorConstant;
        const double a = _angle->getValueAtTime(t) * 3.14159265358979323846 / 180.0;
        const double len = _length->getValueAtTime(t) * (_invert->getValueAtTime(t) ? -1.0 : 1.0);
        s.tx = s.cx + std::cos(a) * len * s.par;
        s.ty = s.cy + std::sin(a) * len;
        s.maxDist = _maxDist->getValueAtTime(t);
        return s;
    }

    Part hit(const State& st, const ovl::Pen& pen, double x, double y) const
    {
        const double r = pen.hitRadius();
        if (st.arrow && pen.screenDist(x, y, st.tx, st.ty) < r) return kTip;
        if (st.maxDist > 0.0 && pen.screenDist(x, y, st.cx + st.maxDist * st.par, st.cy) < r) return kRadius;
        if (pen.screenDist(x, y, st.cx, st.cy) < r) return kCentre;
        return kNone;
    }

    OFX::BooleanParam* _show;
    OFX::Double2DParam* _offset;
    OFX::ChoiceParam* _type;
    OFX::ChoiceParam* _source;
    OFX::DoubleParam* _angle;
    OFX::DoubleParam* _length;
    OFX::BooleanParam* _invert;
    OFX::DoubleParam* _maxDist;
    OFX::Clip* _src;
    Part _drag = kNone, _hover = kNone;
    double _dragX = 0, _dragY = 0, _dragOffX = 0, _dragOffY = 0;
};

class PushOverlayDescriptor : public OFX::DefaultEffectOverlayDescriptor<PushOverlayDescriptor, PushOverlay> {};

} // namespace

mDeclarePluginFactory(push_IT_OFXFactory, {}, ;);

void push_IT_OFXFactory::unload()
{
    cudaReleaseResources();
}

void push_IT_OFXFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(false);              // the spread needs the whole frame
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
    desc.setOverlayInteractDescriptor(new PushOverlayDescriptor);
}

static OFX::ChoiceParamDescriptor* defineChoice(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
                                                OFX::GroupParamDescriptor* group, const char* name,
                                                const char* label, const char* hint)
{
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam(name);
    p->setLabels(label, label, label);
    p->setHint(hint);
    if (group) p->setParent(*group);
    if (page) page->addChild(*p);
    return p;
}

static OFX::DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
                                                OFX::GroupParamDescriptor* group, const char* name, const char* label,
                                                const char* hint, double def, double mn, double mx, double dmn, double dmx)
{
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(mn, mx);
    p->setDisplayRange(dmn, dmx);
    if (group) p->setParent(*group);
    if (page) page->addChild(*p);
    return p;
}

static OFX::BooleanParamDescriptor* defineBool(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
                                               OFX::GroupParamDescriptor* group, const char* name, const char* label,
                                               const char* hint, bool def)
{
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(name);
    p->setLabels(label, label, label);
    p->setHint(hint);
    p->setDefault(def);
    if (group) p->setParent(*group);
    if (page) page->addChild(*p);
    return p;
}

// Section title: a horizontal line followed by a read-only label with the section name.
// Replaces the collapsible groups. `previous` is the last parameter of the section above.
static void defineSection(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
                          OFX::ParamDescriptor* previous, const char* name, const char* title)
{
    if (previous) previous->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintDivider, false);
    OFX::StringParamDescriptor* s = desc.defineStringParam(name);
    s->setLabels(title, title, title);
    s->setStringType(OFX::eStringTypeLabel);
    s->setDefault("");
    s->setIsPersistant(false);
    s->setEvaluateOnChange(false);
    s->setAnimates(false);
    page->addChild(*s);
}

void push_IT_OFXFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    // ---- clips
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->addSupportedComponent(OFX::ePixelComponentAlpha);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    src->setIsMask(false);

    OFX::ClipDescriptor* matte = desc.defineClip(kClipMatte);
    matte->addSupportedComponent(OFX::ePixelComponentRGBA);
    matte->addSupportedComponent(OFX::ePixelComponentRGB);
    matte->addSupportedComponent(OFX::ePixelComponentAlpha);
    matte->setTemporalClipAccess(false);
    matte->setSupportsTiles(false);
    matte->setOptional(true);
    matte->setIsMask(false);

    OFX::ClipDescriptor* vec = desc.defineClip(kClipVector);
    vec->addSupportedComponent(OFX::ePixelComponentRGBA);
    vec->addSupportedComponent(OFX::ePixelComponentRGB);
    vec->setTemporalClipAccess(false);
    vec->setSupportsTiles(false);
    vec->setOptional(true);
    vec->setIsMask(false);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->addSupportedComponent(OFX::ePixelComponentAlpha);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

    // ---- GPU : first line of the panel, outside any section, status on the same line
    OFX::ParamDescriptor* gpuStatus = nullptr;
    {
        OFX::BooleanParamDescriptor* b = defineBool(desc, page, nullptr, kParamUseGPU, "Use GPU (CUDA)",
            "Renders the effect on the NVIDIA GPU. Falls back to the CPU automatically if CUDA is not available.", true);
        b->setEvaluateOnChange(true);
        // Host extension understood by Nuke/Natron: keep the next parameter on this line.
        b->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);

        OFX::StringParamDescriptor* s = desc.defineStringParam(kParamGPUStatus);
        s->setLabels("", "", "");
        s->setHint("GPU used for rendering.");
        s->setStringType(OFX::eStringTypeLabel);
        s->setIsPersistant(false);
        s->setEvaluateOnChange(false);
        s->setAnimates(false);
        std::string info;
        cudaDeviceInfo(info);
        s->setDefault(info);
        page->addChild(*s);
        gpuStatus = s;
    }

    // ---- spread
    defineSection(desc, page, gpuStatus, "sectionSpread", "Spread");
    OFX::GroupParamDescriptor* gSpread = nullptr;
    OFX::ParamDescriptor* last = nullptr;
    {
        OFX::ChoiceParamDescriptor* c = defineChoice(desc, page, gSpread, kParamSpreadType, "Spread Type",
            "Interpolate: smooth fill (push-pull).\nStretch: stretches the nearest edge pixel.\n"
            "Vector Warp: directional stretch (Vector input or constant angle).");
        c->appendOption("Interpolate");
        c->appendOption("Stretch");
        c->appendOption("Vector Warp");
        c->setDefault(kTypeInterpolate);

        c = defineChoice(desc, page, gSpread, kParamSpreadMode, "Spread Mode",
            "Expand: pushes the matte colours outwards.\n"
            "Contract: fills the inside of the matte with the surrounding pixels.");
        c->appendOption("Expand");
        c->appendOption("Contract");
        c->setDefault(kModeExpand);

        defineDouble(desc, page, gSpread, kParamMaxDistance, "Max Distance",
            "Maximum spread distance in pixels (0 = unlimited).", 0, 0, 1e5, 0, 500);
        defineDouble(desc, page, gSpread, kParamFalloff, "Distance Falloff",
            "Softens the end of the spread over this width (pixels).", 0, 0, 1e5, 0, 100);
        defineDouble(desc, page, gSpread, kParamBlur, "Spread Blur",
            "Blur applied to the spread pixels only (pixels).", 0, 0, 1e4, 0, 100);
        last = defineDouble(desc, page, gSpread, kParamSeedThreshold, "Seed Threshold",
            "Minimum matte value for a pixel to be used as a source (Stretch, Max Distance).", 0.5, 0, 1, 0, 1);
    }

    // ---- matte
    defineSection(desc, page, last, "sectionMatte", "Matte");
    OFX::GroupParamDescriptor* gMatte = nullptr;
    {
        OFX::ChoiceParamDescriptor* c = defineChoice(desc, page, gMatte, kParamMatteSource, "Matte Source",
            "Channel that defines the area to keep.\nAuto: alpha of the Matte input if connected, otherwise alpha of Source.");
        c->appendOption("Auto");
        c->appendOption("Source Alpha");
        c->appendOption("Matte Alpha");
        c->appendOption("Matte Luminance");
        c->setDefault(kMatteAuto);

        defineBool(desc, page, gMatte, kParamInvertMatte, "Invert Matte", "Inverts the matte.", false);
        defineBool(desc, page, gMatte, kParamSourcePremult, "Source Premultiplied",
            "The source is premultiplied by its alpha: it is unpremultiplied before the spread.", false);
        defineDouble(desc, page, gMatte, kParamShrink, "Edge Shrink",
            "Erodes the source area by N pixels to ignore contaminated edges.", 0, 0, 1000, 0, 20);
        last = defineDouble(desc, page, gMatte, kParamEdgeSoftness, "Edge Softness",
            "Softens the transition between the original pixels and the spread pixels.", 0, 0, 1000, 0, 20);
    }

    // ---- vector warp
    defineSection(desc, page, last, "sectionVector", "Vector Warp");
    OFX::GroupParamDescriptor* gVec = nullptr;
    {
        OFX::ChoiceParamDescriptor* c = defineChoice(desc, page, gVec, kParamVectorSource, "Vector Source",
            "Vector Input: direction read from the Vector input (pixels).\nConstant: fixed angle and length.");
        c->appendOption("Vector Input");
        c->appendOption("Constant");
        c->setDefault(kVectorConstant);

        defineDouble(desc, page, gVec, kParamAngle, "Angle", "Stretch direction (degrees, 0 = to the right).",
                     0, -360, 360, -180, 180);
        defineDouble(desc, page, gVec, kParamLength, "Length", "Stretch length (pixels).", 200, 0, 1e5, 0, 1000);
        defineDouble(desc, page, gVec, kParamMotionDistance, "Motion Distance",
            "Gain applied to the Vector input. A negative value reverses the direction.", 1, -1e4, 1e4, -10, 10);
        defineDouble(desc, page, gVec, kParamMotionThreshold, "Motion Threshold",
            "Vectors shorter than this threshold (pixels) are ignored.", 0, 0, 1e4, 0, 10);

        OFX::Double2DParamDescriptor* o = desc.defineDouble2DParam(kParamVectorOrigin);
        o->setLabels("Vector Origin", "Vector Origin", "Vector Origin");
        o->setHint("R/G channel values that mean no motion.");
        o->setDefault(0, 0);
        o->setDoubleType(OFX::eDoubleTypePlain);
        page->addChild(*o);

        defineDouble(desc, page, gVec, kParamGainRX, "Red X Gain",   "Red channel contribution to the X motion.", 1, -1e4, 1e4, -2, 2);
        defineDouble(desc, page, gVec, kParamGainRY, "Red Y Gain",   "Red channel contribution to the Y motion.", 0, -1e4, 1e4, -2, 2);
        defineDouble(desc, page, gVec, kParamGainGX, "Green X Gain", "Green channel contribution to the X motion.",  0, -1e4, 1e4, -2, 2);
        defineDouble(desc, page, gVec, kParamGainGY, "Green Y Gain", "Green channel contribution to the Y motion.",  1, -1e4, 1e4, -2, 2);
        defineDouble(desc, page, gVec, kParamGainBX, "Blue X Gain",  "Blue channel contribution to the X motion.",  0, -1e4, 1e4, -2, 2);
        defineDouble(desc, page, gVec, kParamGainBY, "Blue Y Gain",  "Blue channel contribution to the Y motion.",  0, -1e4, 1e4, -2, 2);
        last = defineBool(desc, page, gVec, kParamInvertDirection, "Invert Direction", "Reverses the stretch direction.", false);
    }

    // ---- output
    defineSection(desc, page, last, "sectionOutput", "Output");
    OFX::GroupParamDescriptor* gOut = nullptr;
    {
        OFX::ChoiceParamDescriptor* c = defineChoice(desc, page, gOut, kParamResultOutput, "Result Output",
            "Combined: original image + spread pixels (unpremultiplied RGB).\n"
            "Spread Only: spread pixels only.\n"
            "Premult: Combined result premultiplied by the alpha chosen in Alpha Output.");
        c->appendOption("Combined");
        c->appendOption("Spread Only");
        c->appendOption("Premult");
        c->setDefault(kResultCombined);

        c = defineChoice(desc, page, gOut, kParamAlphaOutput, "Alpha Output",
            "Input: source alpha.\nSpread Matte: matte extended by the spread.\nSolid: alpha set to 1.");
        c->appendOption("Input");
        c->appendOption("Spread Matte");
        c->appendOption("Solid");
        c->setDefault(kAlphaInput);

        // Replaced by Result Output = Premult. Kept hidden so older Nuke scripts still load.
        OFX::BooleanParamDescriptor* legacy = defineBool(desc, page, gOut, kParamPremultOutput, "Premultiply Output",
            "Obsolete: use Result Output = Premult.", false);
        legacy->setIsSecret(true);
        last = defineBool(desc, page, gOut, kParamExtendToFormat, "Extend BBox to Format",
            "Extends the bounding box to the project format so the spread reaches the image edges.", false);
    }

    // ---- viewer
    defineSection(desc, page, last, "sectionViewer", "Viewer");
    {
        OFX::BooleanParamDescriptor* b = defineBool(desc, page, nullptr, kParamShowOverlay, "Show Overlay",
            "Shows the viewer widgets while the panel is open: the Vector Warp arrow (constant vector: drag its tip to "
            "set Angle and Length) and the Max Distance circle (drag its handle). Drag the centre cross to move them.",
            true);
        b->setEvaluateOnChange(false);
        b->setAnimates(false);
        OFX::Double2DParamDescriptor* o = desc.defineDouble2DParam(kParamOverlayOffset);
        o->setLabels("Overlay Offset", "Overlay Offset", "Overlay Offset");
        o->setHint("Position of the viewer widgets, relative to the centre of the format.");
        o->setDefault(0, 0);
        o->setDoubleType(OFX::eDoubleTypePlain);
        o->setEvaluateOnChange(false);
        o->setAnimates(false);
        o->setIsSecret(true);
        page->addChild(*o);
    }

}

OFX::ImageEffect* push_IT_OFXFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new push_IT_OFXPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static push_IT_OFXFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
} // namespace Plugin
} // namespace OFX
