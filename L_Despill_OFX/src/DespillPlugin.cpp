// DespillPlugin.cpp
// OFX front-end of L_Despill_OFX: spill suppression for green / blue / red screens, with
// spill matte processing, spill replace, FG grade and background re-light.
// Based on the L_Despill v05 (Luma Pictures) and DespillMadness gizmos, with their
// inconsistencies fixed. CPU and optional CUDA back-ends.
//
// Inputs : Source, BG (optional), Mask (optional)
#include "ofxsImageEffect.h"
#include "ofxsLog.h"

#include "DespillCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define kPluginName          "L_Despill_OFX"
#define kPluginGrouping      "Keyer"
#define kPluginIdentifier    "fr.ldespill.L_Despill_OFX"
#define kPluginVersionMajor  1
#define kPluginVersionMinor  0

#define kPluginDescription \
    "L_Despill_OFX: spill suppression.\n\n" \
    "The screen channel is limited by a mix of the two other channels (Average or Max), in linear or log space. " \
    "The spill removed gives a spill matte, which can be filtered (median, clean, blur) and drives a grade of " \
    "the despilled area, a grade of the foreground, and the re-light of the edges with a background (BG input).\n\n" \
    "Based on the L_Despill v05 (Luma Pictures) and DespillMadness gizmos.\n" \
    "Optional CUDA GPU acceleration (NVIDIA cards), automatic fallback to the CPU."

static const char* kClipBG   = "BG";
static const char* kClipMask = "Mask";

static const char* kParamUseGPU          = "useGPU";
static const char* kParamGPUStatus       = "gpuStatus";
static const char* kParamScreen          = "screenColor";
static const char* kParamMath            = "despillMath";
static const char* kParamBias            = "bias";
static const char* kParamLimitScale      = "limitScale";
static const char* kParamMathSpace       = "mathColorspace";
static const char* kParamBalanceWhite    = "balanceWhite";
static const char* kParamBalanceBlack    = "balanceBlack";
static const char* kParamRestoreLuma     = "restoreLuminance";
static const char* kParamMatteSpace      = "matteColorspace";
static const char* kParamMatteBlack      = "matteBlackpoint";
static const char* kParamMatteWhite      = "matteWhitepoint";
static const char* kParamMatteGamma      = "matteGamma";
static const char* kParamMedian          = "medianSize";
static const char* kParamClean           = "cleanSize";
static const char* kParamMatteBlur       = "matteBlur";
static const char* kParamReplaceSpace    = "replaceColorspace";
static const char* kParamBGMode          = "bgMode";
static const char* kParamBGSpace         = "bgColorspace";
static const char* kParamBGBlack         = "bgBlackpoint";
static const char* kParamBGWhite         = "bgWhitepoint";
static const char* kParamBGGamma         = "bgGamma";
static const char* kParamBGSaturation    = "bgSaturation";
static const char* kParamBGBlackClamp    = "bgBlackClamp";
static const char* kParamBGWhiteClamp    = "bgWhiteClamp";
static const char* kParamAlphaOut        = "alphaOutput";
static const char* kParamInvertMask      = "invertMask";
static const char* kParamMix             = "mix";

// Grade parameter sets: prefix + suffix (spill replace "replace*", FG grade "fg*")
static const char* kGradeSuffix[7] = { "Blackpoint", "Whitepoint", "Lift", "Gain", "Multiply", "Offset", "Gamma" };
static const char* kGradeLabel[7]  = { "blackpoint", "whitepoint", "lift", "gain", "multiply", "offset", "gamma" };

// Layout hints (host extension, not in the base OFX headers).
static const char* kPropLayoutHint = "OfxParamPropLayoutHint";
static const int   kLayoutHintNoNewLine = 2;
static const int   kLayoutHintDivider   = 1;

static const int   kMaxMedianSize = 10;       // also the GPU limit

using namespace ldespill;

// ============================================================ helpers
namespace {

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

inline int componentCount(OFX::PixelComponentEnum c)
{
    switch (c) {
    case OFX::ePixelComponentRGBA:  return 4;
    case OFX::ePixelComponentRGB:   return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default:                        return 0;
    }
}

enum RowMode { kRowRGBA = 4, kRowRGB = 3, kRowMatte = 1 };

// Rows [r0, r1) of `rect` (0 = bottom row) of `img`, written densely at `out`:
// RGBA (RGB images get alpha 1), RGB, or a matte (alpha; luminance for RGB images;
// clamped to [0, 1], optionally inverted). Outside the image: black, or (repeatEdges) the
// edge pixels repeated, as Nuke does outside a bounding box. Thread-safe (reads only).
void readRows(const OFX::Image* img, const OfxRectI& rect, int r0, int r1, bool repeatEdges, RowMode mode,
              bool invert, float* out)
{
    const int w = rect.x2 - rect.x1, stride = (int)mode;
    const int nc = img ? componentCount(img->getPixelComponents()) : 0;
    const OfxRectI bd = img ? img->getBounds() : OfxRectI{ 0, 0, 0, 0 };
    const bool empty = nc == 0 || bd.x2 <= bd.x1 || bd.y2 <= bd.y1;
    auto put = [&](const float* px, float* o) {
        float c[4] = { 0.f, 0.f, 0.f, 0.f };
        if (px) {
            if (nc == 4)      { c[0] = px[0]; c[1] = px[1]; c[2] = px[2]; c[3] = px[3]; }
            else if (nc == 3) { c[0] = px[0]; c[1] = px[1]; c[2] = px[2]; c[3] = 1.f; }
            else              { c[3] = px[0]; }
        }
        if (mode == kRowMatte) {
            float m = nc == 3 ? 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2] : c[3];
            m = std::min(1.f, std::max(0.f, m));
            o[0] = invert ? 1.f - m : m;
            return;
        }
        for (int k = 0; k < stride; ++k) o[k] = c[k];
    };
    for (int r = r0; r < r1; ++r) {
        float* o = out + (size_t)(r - r0) * w * stride;
        const int y = rect.y1 + r;
        const float* row = nullptr;
        if (!empty) {
            const int sy = repeatEdges ? std::min(std::max(y, bd.y1), bd.y2 - 1) : y;
            if (sy >= bd.y1 && sy < bd.y2) row = static_cast<const float*>(img->getPixelAddress(bd.x1, sy));
        }
        // fast paths: RGBA image, row fully inside the image: plain copy (RGBA) or tight loop (RGB)
        if (row && nc == 4 && rect.x1 >= bd.x1 && rect.x2 <= bd.x2 && (mode == kRowRGBA || mode == kRowRGB)) {
            const float* s = row + (size_t)(rect.x1 - bd.x1) * 4;
            if (mode == kRowRGBA) {
                std::memcpy(o, s, sizeof(float) * 4 * (size_t)w);
            } else {
                for (int x = 0; x < w; ++x, s += 4, o += 3) { o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; }
            }
            continue;
        }
        for (int x = rect.x1; x < rect.x2; ++x, o += stride) {
            const int sx = repeatEdges ? std::min(std::max(x, bd.x1), bd.x2 - 1) : x;
            put(row && sx >= bd.x1 && sx < bd.x2 ? row + (size_t)(sx - bd.x1) * nc : nullptr, o);
        }
    }
}

// Dense RGBA rows [r0, r1) of `win` into the output image.
void writeRows(OFX::Image* img, const OfxRectI& win, int r0, int r1, const float* in)
{
    const int w = win.x2 - win.x1;
    const int nc = componentCount(img->getPixelComponents());
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

inline int choiceAt(OFX::ChoiceParam* p, double t)
{
    int v = 0;
    p->getValueAtTime(t, v);
    return v;
}

} // namespace

// ================================================================== plugin
class LDespillPlugin : public OFX::ImageEffect
{
public:
    explicit LDespillPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;
    void changedClip(const OFX::InstanceChangedArgs& args, const std::string& clipName) override;

private:
    Params paramsAt(double t, double renderScale);
    void readGrade(OFX::RGBParam* const* g, OFX::DoubleParam* sat, double t, Grade& out);
    void updateParamsState();

    OFX::Clip* _dstClip;
    OFX::Clip* _srcClip;
    OFX::Clip* _bgClip;
    OFX::Clip* _maskClip;

    OFX::BooleanParam* _useGPU;
    OFX::ChoiceParam*  _screen;
    OFX::ChoiceParam*  _math;
    OFX::DoubleParam*  _bias;
    OFX::DoubleParam*  _limitScale;
    OFX::ChoiceParam*  _mathSpace;
    OFX::DoubleParam*  _balanceWhite;
    OFX::DoubleParam*  _balanceBlack;
    OFX::DoubleParam*  _restoreLuma;
    OFX::ChoiceParam*  _matteSpace;
    OFX::DoubleParam*  _matteBlack;
    OFX::DoubleParam*  _matteWhite;
    OFX::DoubleParam*  _matteGamma;
    OFX::IntParam*     _median;
    OFX::IntParam*     _clean;
    OFX::DoubleParam*  _matteBlur;
    OFX::ChoiceParam*  _replaceSpace;
    OFX::RGBParam*     _replace[7];
    OFX::DoubleParam*  _replaceSat;
    OFX::RGBParam*     _fg[7];
    OFX::DoubleParam*  _fgSat;
    OFX::ChoiceParam*  _bgMode;
    OFX::ChoiceParam*  _bgSpace;
    OFX::DoubleParam*  _bgBlack;
    OFX::DoubleParam*  _bgWhite;
    OFX::DoubleParam*  _bgGamma;
    OFX::DoubleParam*  _bgSat;
    OFX::BooleanParam* _bgBlackClamp;
    OFX::BooleanParam* _bgWhiteClamp;
    OFX::ChoiceParam*  _alphaOut;
    OFX::BooleanParam* _invertMask;
    OFX::DoubleParam*  _mix;
};

LDespillPlugin::LDespillPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _dstClip  = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip  = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _bgClip   = fetchClip(kClipBG);
    _maskClip = fetchClip(kClipMask);

    _useGPU       = fetchBooleanParam(kParamUseGPU);
    _screen       = fetchChoiceParam(kParamScreen);
    _math         = fetchChoiceParam(kParamMath);
    _bias         = fetchDoubleParam(kParamBias);
    _limitScale   = fetchDoubleParam(kParamLimitScale);
    _mathSpace    = fetchChoiceParam(kParamMathSpace);
    _balanceWhite = fetchDoubleParam(kParamBalanceWhite);
    _balanceBlack = fetchDoubleParam(kParamBalanceBlack);
    _restoreLuma  = fetchDoubleParam(kParamRestoreLuma);
    _matteSpace   = fetchChoiceParam(kParamMatteSpace);
    _matteBlack   = fetchDoubleParam(kParamMatteBlack);
    _matteWhite   = fetchDoubleParam(kParamMatteWhite);
    _matteGamma   = fetchDoubleParam(kParamMatteGamma);
    _median       = fetchIntParam(kParamMedian);
    _clean        = fetchIntParam(kParamClean);
    _matteBlur    = fetchDoubleParam(kParamMatteBlur);
    _replaceSpace = fetchChoiceParam(kParamReplaceSpace);
    for (int i = 0; i < 7; ++i) {
        _replace[i] = fetchRGBParam(std::string("replace") + kGradeSuffix[i]);
        _fg[i]      = fetchRGBParam(std::string("fg") + kGradeSuffix[i]);
    }
    _replaceSat   = fetchDoubleParam("replaceSaturation");
    _fgSat        = fetchDoubleParam("fgSaturation");
    _bgMode       = fetchChoiceParam(kParamBGMode);
    _bgSpace      = fetchChoiceParam(kParamBGSpace);
    _bgBlack      = fetchDoubleParam(kParamBGBlack);
    _bgWhite      = fetchDoubleParam(kParamBGWhite);
    _bgGamma      = fetchDoubleParam(kParamBGGamma);
    _bgSat        = fetchDoubleParam(kParamBGSaturation);
    _bgBlackClamp = fetchBooleanParam(kParamBGBlackClamp);
    _bgWhiteClamp = fetchBooleanParam(kParamBGWhiteClamp);
    _alphaOut     = fetchChoiceParam(kParamAlphaOut);
    _invertMask   = fetchBooleanParam(kParamInvertMask);
    _mix          = fetchDoubleParam(kParamMix);
    updateParamsState();
}

void LDespillPlugin::updateParamsState()
{
    _useGPU->setEnabled(cudaIsCompiledIn());
    int mode = 0;
    _bgMode->getValue(mode);
    const bool bg = _bgClip->isConnected();
    _bgMode->setEnabled(bg);
    const bool add = bg && mode == kBGAdd;
    _bgSpace->setEnabled(add);
    _bgBlack->setEnabled(add);
    _bgWhite->setEnabled(add);
    _bgGamma->setEnabled(add);
    _bgSat->setEnabled(add);
    _bgBlackClamp->setEnabled(add);
    _bgWhiteClamp->setEnabled(add);
}

void LDespillPlugin::changedParam(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kParamBGMode) updateParamsState();
}

void LDespillPlugin::changedClip(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kClipBG) updateParamsState();
}

void LDespillPlugin::readGrade(OFX::RGBParam* const* g, OFX::DoubleParam* sat, double t, Grade& out)
{
    float* dst[7] = { out.blackpoint, out.whitepoint, out.lift, out.gain, out.multiply, out.offset, out.gamma };
    for (int i = 0; i < 7; ++i) {
        double r, gg, b;
        g[i]->getValueAtTime(t, r, gg, b);
        dst[i][0] = (float)r; dst[i][1] = (float)gg; dst[i][2] = (float)b;
    }
    out.saturation = (float)sat->getValueAtTime(t);
    out.blackClamp = out.whiteClamp = false;
}

Params LDespillPlugin::paramsAt(double t, double scale)
{
    Params p;
    p.screen = choiceAt(_screen, t);
    p.math = choiceAt(_math, t);
    p.bias = (float)_bias->getValueAtTime(t);
    p.limitScale = (float)_limitScale->getValueAtTime(t);
    p.mathLog = choiceAt(_mathSpace, t) == 0;
    p.balanceWhite = (float)_balanceWhite->getValueAtTime(t);
    p.balanceBlack = (float)_balanceBlack->getValueAtTime(t);
    p.restoreLuminance = (float)_restoreLuma->getValueAtTime(t);

    p.matteSpace = choiceAt(_matteSpace, t);
    p.matteBlack = (float)_matteBlack->getValueAtTime(t);
    p.matteWhite = (float)_matteWhite->getValueAtTime(t);
    p.matteGamma = (float)_matteGamma->getValueAtTime(t);
    p.medianRadius = std::min(kMaxMedianSize, std::max(0, (int)std::lround(_median->getValueAtTime(t) * scale)));
    p.cleanRadius = (int)std::lround(_clean->getValueAtTime(t) * scale);
    p.matteBlur = (float)(std::max(0.0, _matteBlur->getValueAtTime(t)) * scale);

    p.replaceLog = choiceAt(_replaceSpace, t) == 0;
    readGrade(_replace, _replaceSat, t, p.replace);
    readGrade(_fg, _fgSat, t, p.fg);

    p.bgMode = choiceAt(_bgMode, t);
    p.bgLog = choiceAt(_bgSpace, t) == 0;
    const float bb = (float)_bgBlack->getValueAtTime(t), bw = (float)_bgWhite->getValueAtTime(t);
    const float bgam = (float)_bgGamma->getValueAtTime(t);
    for (int k = 0; k < 3; ++k) { p.bg.blackpoint[k] = bb; p.bg.whitepoint[k] = bw; p.bg.gamma[k] = bgam; }
    p.bg.saturation = (float)_bgSat->getValueAtTime(t);
    p.bg.blackClamp = _bgBlackClamp->getValueAtTime(t);
    p.bg.whiteClamp = _bgWhiteClamp->getValueAtTime(t);

    p.alphaOut = choiceAt(_alphaOut, t);
    p.mix = (float)_mix->getValueAtTime(t);
    return p;
}

bool LDespillPlugin::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if (!_srcClip->isConnected()) return false;
    rod = _srcClip->getRegionOfDefinition(args.time);
    return true;
}

void LDespillPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois)
{
    const double e = spatialMargin(paramsAt(args.time, 1.0));
    const OfxRectD& r = args.regionOfInterest;
    rois.setRegionOfInterest(*_srcClip, { r.x1 - e, r.y1 - e, r.x2 + e, r.y2 + e });
    if (_bgClip->isConnected()) rois.setRegionOfInterest(*_bgClip, r);
    if (_maskClip->isConnected()) rois.setRegionOfInterest(*_maskClip, r);
}

void LDespillPlugin::render(const OFX::RenderArguments& args)
{
    const double t = args.time;
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(t));
    if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (dst->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    const OfxRectI win = dst->getBounds();
    const int ww = win.x2 - win.x1, wh = win.y2 - win.y1;
    if (ww <= 0 || wh <= 0) return;

    const double scale = 0.5 * (args.renderScale.x + args.renderScale.y);
    Params p = paramsAt(t, scale);

    std::unique_ptr<const OFX::Image> src(_srcClip->isConnected() ? _srcClip->fetchImage(t) : nullptr);
    std::unique_ptr<const OFX::Image> bg(_bgClip->isConnected() && p.bgMode != kBGOff ? _bgClip->fetchImage(t) : nullptr);
    std::unique_ptr<const OFX::Image> mask(_maskClip->isConnected() ? _maskClip->fetchImage(t) : nullptr);
    for (const OFX::Image* im : { src.get(), bg.get(), mask.get() })
        if (im && im->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    const bool invertMask = _invertMask->getValueAtTime(t);
    p.hasBG = bg != nullptr;

    // working buffer: output window + the margin of the spatial filters (source edges repeated)
    const int margin = spatialMargin(p);
    const OfxRectI buf = { win.x1 - margin, win.y1 - margin, win.x2 + margin, win.y2 + margin };
    p.bufW = buf.x2 - buf.x1; p.bufH = buf.y2 - buf.y1;
    p.winX = margin; p.winY = margin; p.winW = ww; p.winH = wh;

    // ---- GPU: rows go straight from the images into the transfer memory and back
    if (_useGPU->getValueAtTime(t) && cudaIsCompiledIn()) {
        const RowReader rSrc = [&](int r0, int r1, float* o) { readRows(src.get(), buf, r0, r1, true, kRowRGBA, false, o); };
        const RowReader rBG = [&](int r0, int r1, float* o) { readRows(bg.get(), win, r0, r1, false, kRowRGB, false, o); };
        const RowReader rMask = [&](int r0, int r1, float* o) { readRows(mask.get(), win, r0, r1, false, kRowMatte, invertMask, o); };
        const RowWriter wOut = [&](int r0, int r1, const float* i) { writeRows(dst.get(), win, r0, r1, i); };
        std::string err;
        if (processCUDARows(p, rSrc, bg ? &rBG : nullptr, mask ? &rMask : nullptr, wOut, err)) return;
        OFX::Log::warning(true, "L_Despill_OFX: GPU render failed (%s), falling back to CPU", err.c_str());
    }

    // ---- CPU
    const size_t NB = (size_t)p.bufW * p.bufH, NW = (size_t)ww * wh;
    std::vector<float> srcBuf(4 * NB), bgBuf, maskBuf, outBuf(4 * NW);
    parallelRanges(p.bufH, [&](int r0, int r1) {
        readRows(src.get(), buf, r0, r1, true, kRowRGBA, false, &srcBuf[4 * (size_t)r0 * p.bufW]);
    });
    if (bg) {
        bgBuf.resize(3 * NW);
        parallelRanges(wh, [&](int r0, int r1) { readRows(bg.get(), win, r0, r1, false, kRowRGB, false, &bgBuf[3 * (size_t)r0 * ww]); });
    }
    if (mask) {
        maskBuf.resize(NW);
        parallelRanges(wh, [&](int r0, int r1) { readRows(mask.get(), win, r0, r1, false, kRowMatte, invertMask, &maskBuf[(size_t)r0 * ww]); });
    }
    processCPU(p, srcBuf.data(), bg ? bgBuf.data() : nullptr, mask ? maskBuf.data() : nullptr, outBuf.data());
    parallelRanges(wh, [&](int r0, int r1) { writeRows(dst.get(), win, r0, r1, &outBuf[4 * (size_t)r0 * ww]); });
}

// ================================================================= factory
mDeclarePluginFactory(LDespillFactory, {}, ;);

void LDespillFactory::unload() { cudaReleaseResources(); }

void LDespillFactory::describe(OFX::ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
}

namespace {

// Tab the parameters being defined belong to. Nuke ignores OFX pages, but shows a group
// flagged "FnOfxParamPropGroupIsTab" as a tab of the node panel.
OFX::GroupParamDescriptor* gTab = nullptr;

void beginTab(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* name, const char* label)
{
    OFX::GroupParamDescriptor* g = d.defineGroupParam(name);
    g->setLabels(label, label, label);
    g->getPropertySet().propSetInt("FnOfxParamPropGroupIsTab", 1, false);
    page->addChild(*g);
    gTab = g;
}

template <class D>
D* finish(D* p, OFX::PageParamDescriptor* page, const char* label, const char* hint)
{
    p->setLabels(label, label, label);
    p->setHint(hint);
    if (gTab) p->setParent(*gTab);
    page->addChild(*p);
    return p;
}

OFX::DoubleParamDescriptor* dbl(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* name,
                                const char* label, const char* hint, double def, double mn, double mx, double dmn, double dmx)
{
    OFX::DoubleParamDescriptor* p = finish(d.defineDoubleParam(name), page, label, hint);
    p->setDefault(def); p->setRange(mn, mx); p->setDisplayRange(dmn, dmx);
    return p;
}

OFX::BooleanParamDescriptor* boolean(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* name,
                                     const char* label, const char* hint, bool def)
{
    OFX::BooleanParamDescriptor* p = finish(d.defineBooleanParam(name), page, label, hint);
    p->setDefault(def);
    return p;
}

OFX::ChoiceParamDescriptor* choice(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* name,
                                   const char* label, const char* hint, std::initializer_list<const char*> options, int def)
{
    OFX::ChoiceParamDescriptor* p = finish(d.defineChoiceParam(name), page, label, hint);
    for (const char* o : options) p->appendOption(o);
    p->setDefault(def);
    return p;
}

OFX::IntParamDescriptor* integer(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* name,
                                 const char* label, const char* hint, int def, int mn, int mx)
{
    OFX::IntParamDescriptor* p = finish(d.defineIntParam(name), page, label, hint);
    p->setDefault(def); p->setRange(mn, mx); p->setDisplayRange(mn, mx);
    return p;
}

// Nuke Grade knobs (RGB) + saturation; returns the last parameter (for the divider).
OFX::ParamDescriptor* gradeParams(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const char* prefix,
                                  const char* labelPrefix, const char* what)
{
    static const double defs[7] = { 0, 1, 0, 1, 1, 0, 1 };
    static const double dmin[7] = { -1, 0, -1, 0, 0, -1, 0.2 };
    static const double dmax[7] = { 1, 4, 1, 4, 4, 1, 5 };
    static const char* hints[7] = {
        "Black point: this colour becomes black.",
        "White point: this colour becomes white.",
        "Lift: black is raised to this colour.",
        "Gain: white is multiplied to this colour.",
        "Multiply: multiplies the colour.",
        "Offset: added to the colour.",
        "Gamma: applied after the other settings." };
    for (int i = 0; i < 7; ++i) {
        const std::string name = std::string(prefix) + kGradeSuffix[i];
        const std::string label = std::string(labelPrefix) + kGradeLabel[i];
        const std::string hint = std::string(what) + " - " + hints[i];
        OFX::RGBParamDescriptor* p = d.defineRGBParam(name);
        p->setLabels(label, label, label);
        p->setHint(hint);
        p->setDefault(defs[i], defs[i], defs[i]);
        p->setDisplayRange(dmin[i], dmin[i], dmin[i], dmax[i], dmax[i], dmax[i]);
        if (gTab) p->setParent(*gTab);
        page->addChild(*p);
    }
    const std::string sname = std::string(prefix) + "Saturation";
    const std::string slabel = std::string(labelPrefix) + "saturation";
    const std::string shint = std::string(what) + " - Saturation (Rec. 709 luminance), after the grade.";
    OFX::DoubleParamDescriptor* s = d.defineDoubleParam(sname);
    s->setLabels(slabel, slabel, slabel);
    s->setHint(shint);
    s->setDefault(1); s->setRange(-10, 10); s->setDisplayRange(0, 4);
    if (gTab) s->setParent(*gTab);
    page->addChild(*s);
    return s;
}

} // namespace

void LDespillFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* bg = desc.defineClip(kClipBG);
    bg->addSupportedComponent(OFX::ePixelComponentRGBA);
    bg->addSupportedComponent(OFX::ePixelComponentRGB);
    bg->setOptional(true);
    bg->setSupportsTiles(false);

    OFX::ClipDescriptor* mask = desc.defineClip(kClipMask);
    mask->addSupportedComponent(OFX::ePixelComponentAlpha);
    mask->addSupportedComponent(OFX::ePixelComponentRGBA);
    mask->setOptional(true);
    mask->setSupportsTiles(false);
    mask->setIsMask(true);

    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");
    auto divider = [](OFX::ParamDescriptor* p) { p->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintDivider, false); };

    // ================================================================ Despill tab
    beginTab(desc, page, "tabDespill", "Despill");
    // GPU: first line, GPU name on the same line
    {
        OFX::BooleanParamDescriptor* b = boolean(desc, page, kParamUseGPU, "Use GPU (CUDA)",
            "Renders on the NVIDIA GPU. Falls back to the CPU automatically if CUDA is not available.", true);
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
        divider(s);
        s->setParent(*gTab);
        page->addChild(*s);
    }
    choice(desc, page, kParamScreen, "Screen Color", "Colour of the screen whose spill is removed.",
           { "Green", "Blue", "Red" }, kScreenGreen);
    choice(desc, page, kParamMath, "Despill Math",
           "How the limit of the screen channel is computed from the two other channels.\n"
           "Average: weighted average of the two (classic, soft).\n"
           "Max: the larger of the two (gentler on colours close to the screen).",
           { "Average", "Max" }, kMathAverage);
    dbl(desc, page, kParamBias, "Bias",
        "Weight between the two other channels in the limit.\n"
        "Green screen: -1 = red limits green, +1 = blue limits green.\n"
        "Blue screen: -1 = red limits blue, +1 = green limits blue.\n"
        "Red screen: -1 = green limits red, +1 = blue limits red.\n"
        "0 = equal weights. -0.33 = double red average (2R + B) / 3, +0.33 = double blue/green average.\n"
        "L_Despill 'Fine Tune' t (Average, Max) corresponds to Bias = -2t.",
        0, -1, 1, -1, 1);
    dbl(desc, page, kParamLimitScale, "Limit Scale",
        "Multiplies the limit. Below 1: stronger despill; above 1: lighter despill.\n"
        "(L_Despill 'Fine Tune' + 1 for the Average2 / Limits modes.)",
        1, 0, 4, 0.5, 1.5);
    choice(desc, page, kParamMathSpace, "Math Colorspace",
           "Space in which the despill is computed. Log (Cineon) usually gives a more natural result on bright "
           "and dark areas. The image stays linear in and out.",
           { "Log", "Linear" }, 0);
    dbl(desc, page, kParamBalanceWhite, "Balance White",
        "White point of the screen channel before the despill (reverted afterwards). Use it when the whole plate "
        "has a cast of the screen colour, so that the despill focuses on the screen itself.",
        1, -1e3, 1e3, -2, 4);
    dbl(desc, page, kParamBalanceBlack, "Balance Black",
        "Black point of the screen channel before the despill (reverted afterwards).", 0, -1e3, 1e3, -0.01, 0.01);
    divider(dbl(desc, page, kParamRestoreLuma, "Restore Luminance",
        "Adds back the luminance removed with the spill, so that despilled areas do not get darker. "
        "0 = off, 1 = full.",
        0, 0, 1, 0, 1));
    choice(desc, page, kParamAlphaOut, "Alpha Output", "Alpha of the result.", { "Source Alpha", "Spill Matte" }, kAlphaSource);
    boolean(desc, page, kParamInvertMask, "Invert Mask", "Inverts the Mask input.", false);
    divider(dbl(desc, page, kParamMix, "Mix", "Effect mix.", 1, 0, 1, 0, 1));
    {
        OFX::StringParamDescriptor* s = desc.defineStringParam("creditsLine1");
        s->setLabels("", "", "");
        s->setStringType(OFX::eStringTypeLabel);
        s->setDefault("Based on L_Despill v05 by Justin Johnson (Luma Pictures)");
        s->setIsPersistant(false);
        s->setEvaluateOnChange(false);
        s->setAnimates(false);
        s->setParent(*gTab);
        page->addChild(*s);
    }

    // ================================================================ Spill Replace tab
    beginTab(desc, page, "tabSpillReplace", "Spill Replace");
    choice(desc, page, kParamReplaceSpace, "Colorspace", "Space in which the spill replace grade is applied.",
           { "Log", "Linear" }, 0);
    gradeParams(desc, page, "replace", "", "Spill replace: grade of the despilled area (inside the spill matte)");

    // ================================================================ Spill Matte tab
    beginTab(desc, page, "tabSpillMatte", "Spill Matte");
    choice(desc, page, kParamMatteSpace, "Colorspace",
           "Space in which the spill matte (amount of spill removed) is measured. Linear gives a matte proportional "
           "to the colour removed; in Log the values are smaller (raise the whitepoint).",
           { "Same as Despill", "Linear", "Log" }, kMatteLinear);
    dbl(desc, page, kParamMatteWhite, "whitepoint", "Spill amount that becomes 1 in the spill matte.", 1, -10, 10, 0, 1);
    dbl(desc, page, kParamMatteBlack, "blackpoint", "Spill amount that becomes 0 in the spill matte.", 0, -10, 10, 0, 1);
    divider(dbl(desc, page, kParamMatteGamma, "gamma", "Gamma of the spill matte.", 1, 0.01, 10, 0.2, 4));
    integer(desc, page, kParamMedian, "Median Size",
            "Median filter radius (pixels) on the spill matte: removes noise and grain. 0 = off.", 0, 0, kMaxMedianSize);
    integer(desc, page, kParamClean, "Spill Matte Clean",
            "Cleans the spill matte (pixels).\nPositive: fills small holes (dilate then erode).\n"
            "Negative: removes small specks (erode then dilate). 0 = off.", 0, -50, 50);
    dbl(desc, page, kParamMatteBlur, "Spill Matte Blur", "Blur of the spill matte (pixels).", 0, 0, 1000, 0, 100);

    // ================================================================ MultPlus tab
    beginTab(desc, page, "tabMultPlus", "MultPlus");
    choice(desc, page, kParamBGMode, "BG Mode",
           "Use of the BG input inside the spill matte.\n"
           "Add: the graded BG is multiplied by the spill matte and added (re-lights the edges with the background).\n"
           "Keymix: the BG replaces the despilled image inside the spill matte (grade not used).\n"
           "Off: BG input ignored.",
           { "Off", "Add", "Keymix" }, kBGAdd);
    choice(desc, page, kParamBGSpace, "Colorspace", "Space of the Add mode (grade + addition).", { "Log", "Linear" }, 0);
    dbl(desc, page, kParamBGBlack, "blackpoint", "Black point of the BG before it is added.", 0.35, -10, 10, 0, 1);
    dbl(desc, page, kParamBGWhite, "whitepoint", "White point of the BG before it is added.", 0.5, -10, 10, 0, 1);
    dbl(desc, page, kParamBGGamma, "gamma", "Gamma of the BG before it is added.", 1, 0.01, 10, 0.2, 4);
    dbl(desc, page, kParamBGSaturation, "saturation", "Saturation of the BG before it is added.", 0.8, -10, 10, 0, 2);
    {
        OFX::BooleanParamDescriptor* b = boolean(desc, page, kParamBGBlackClamp, "black clamp", "Clamps the graded BG at 0.", false);
        b->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
    }
    boolean(desc, page, kParamBGWhiteClamp, "white clamp", "Clamps the graded BG at 1.", false);

    // ================================================================ FG Grade tab
    beginTab(desc, page, "tabFGGrade", "FG Grade");
    gradeParams(desc, page, "fg", "", "FG grade: grade of the foreground (outside the spill matte)");
    gTab = nullptr;
}

OFX::ImageEffect* LDespillFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new LDespillPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static LDespillFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
} // namespace Plugin
} // namespace OFX
