// DespillITPlugin.cpp
// OFX front-end of despill_IT_OFX: spatially aware despill.
// The spill is measured against the local screen colour, its removal is full near the screen
// and fades towards the inside of the subject, protected hues are kept, the colour of the
// semi-transparent edges is rebuilt from the opaque core, and the removed screen light can be
// replaced by the blurred background, with an optional light wrap. CPU and CUDA back-ends.
//
// Inputs : Source, Key (optional), CleanPlate (optional), BG (optional), Protect (optional),
//          Mask (optional)
#include "ofxsImageEffect.h"
#include "ofxsLog.h"

#include "DespillITCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define kPluginName          "despill_IT_OFX"
#define kPluginGrouping      "Keyer"
#define kPluginIdentifier    "fr.despillit.despill_IT_OFX"
#define kPluginVersionMajor  1
#define kPluginVersionMinor  0

#define kPluginDescription \
    "despill_IT_OFX: spatially aware despill.\n\n" \
    "The spill is measured against the local screen colour (estimated from the plate or read from a clean plate). " \
    "Its removal is full near the screen and fades towards the inside of the subject, so that colours far from " \
    "the edges (green eyes, teal clothes) are kept; hues can also be protected. The colour of semi-transparent " \
    "edges is rebuilt from the opaque core of the subject, and the removed screen light can be replaced by the " \
    "blurred background, with an optional light wrap.\n\n" \
    "Key input: alpha of a keyer (optional, an internal key is used otherwise).\n" \
    "Optional CUDA GPU acceleration (NVIDIA cards), automatic fallback to the CPU."

static const char* kClipKey     = "Key";
static const char* kClipClean   = "CleanPlate";
static const char* kClipBG      = "BG";
static const char* kClipProtect = "Protect";
static const char* kClipMask    = "Mask";

static const char* kParamUseGPU       = "useGPU";
static const char* kParamGPUStatus    = "gpuStatus";
static const char* kParamScreenColor  = "screenColor";
static const char* kParamScreenSource = "screenSource";
static const char* kParamMath         = "despillMath";
static const char* kParamBias         = "bias";
static const char* kParamLimitScale   = "limitScale";
static const char* kParamKeySource    = "keySource";
static const char* kParamKeyLow       = "keyLow";
static const char* kParamKeyHigh      = "keyHigh";
static const char* kParamScreenBelow  = "screenBelow";
static const char* kParamCoreAbove    = "coreAbove";
static const char* kParamStrength     = "strength";
static const char* kParamSpatial      = "spatial";
static const char* kParamEdgeWidth    = "edgeWidth";
static const char* kParamEdgeFalloff  = "edgeFalloff";
static const char* kParamInterior     = "interiorStrength";
static const char* kParamProtectHue   = "protectHue";
static const char* kParamProtectRange = "protectRange";
static const char* kParamProtectSoft  = "protectSoftness";
static const char* kParamProtectAmt   = "protectAmount";
static const char* kParamMode         = "despillMode";
static const char* kParamEdgeRestore  = "edgeRestore";
static const char* kParamSpillLight   = "spillLight";
static const char* kParamBGGain       = "bgGain";
static const char* kParamBGBlur       = "bgBlur";
static const char* kParamWrap         = "lightWrap";
static const char* kParamOutput       = "output";
static const char* kParamAlphaOut     = "alphaOutput";
static const char* kParamInvertMask   = "invertMask";
static const char* kParamMix          = "mix";

static const char* kPropLayoutHint = "OfxParamPropLayoutHint";
static const int   kLayoutHintNoNewLine = 2;
static const int   kLayoutHintDivider   = 1;

enum KeyChoice { kKeyChoiceAuto = 0, kKeyChoiceInternal = 1 };

using namespace dspit;

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
class DespillITPlugin : public OFX::ImageEffect
{
public:
    explicit DespillITPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;
    void changedClip(const OFX::InstanceChangedArgs& args, const std::string& clipName) override;

private:
    Params paramsAt(double t, double renderScale);
    void updateParamsState();

    OFX::Clip *_dstClip, *_srcClip, *_keyClip, *_cleanClip, *_bgClip, *_protectClip, *_maskClip;

    OFX::BooleanParam* _useGPU;
    OFX::RGBParam*     _screenColor;
    OFX::ChoiceParam*  _screenSource;
    OFX::ChoiceParam*  _math;
    OFX::DoubleParam*  _bias;
    OFX::DoubleParam*  _limitScale;
    OFX::ChoiceParam*  _keySource;
    OFX::DoubleParam*  _keyLow;
    OFX::DoubleParam*  _keyHigh;
    OFX::DoubleParam*  _screenBelow;
    OFX::DoubleParam*  _coreAbove;
    OFX::DoubleParam*  _strength;
    OFX::BooleanParam* _spatial;
    OFX::DoubleParam*  _edgeWidth;
    OFX::DoubleParam*  _edgeFalloff;
    OFX::DoubleParam*  _interior;
    OFX::DoubleParam*  _protectHue;
    OFX::DoubleParam*  _protectRange;
    OFX::DoubleParam*  _protectSoft;
    OFX::DoubleParam*  _protectAmt;
    OFX::ChoiceParam*  _mode;
    OFX::DoubleParam*  _edgeRestore;
    OFX::ChoiceParam*  _spillLight;
    OFX::DoubleParam*  _bgGain;
    OFX::DoubleParam*  _bgBlur;
    OFX::DoubleParam*  _wrap;
    OFX::ChoiceParam*  _output;
    OFX::ChoiceParam*  _alphaOut;
    OFX::BooleanParam* _invertMask;
    OFX::DoubleParam*  _mix;
};

DespillITPlugin::DespillITPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _dstClip     = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip     = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _keyClip     = fetchClip(kClipKey);
    _cleanClip   = fetchClip(kClipClean);
    _bgClip      = fetchClip(kClipBG);
    _protectClip = fetchClip(kClipProtect);
    _maskClip    = fetchClip(kClipMask);

    _useGPU       = fetchBooleanParam(kParamUseGPU);
    _screenColor  = fetchRGBParam(kParamScreenColor);
    _screenSource = fetchChoiceParam(kParamScreenSource);
    _math         = fetchChoiceParam(kParamMath);
    _bias         = fetchDoubleParam(kParamBias);
    _limitScale   = fetchDoubleParam(kParamLimitScale);
    _keySource    = fetchChoiceParam(kParamKeySource);
    _keyLow       = fetchDoubleParam(kParamKeyLow);
    _keyHigh      = fetchDoubleParam(kParamKeyHigh);
    _screenBelow  = fetchDoubleParam(kParamScreenBelow);
    _coreAbove    = fetchDoubleParam(kParamCoreAbove);
    _strength     = fetchDoubleParam(kParamStrength);
    _spatial      = fetchBooleanParam(kParamSpatial);
    _edgeWidth    = fetchDoubleParam(kParamEdgeWidth);
    _edgeFalloff  = fetchDoubleParam(kParamEdgeFalloff);
    _interior     = fetchDoubleParam(kParamInterior);
    _protectHue   = fetchDoubleParam(kParamProtectHue);
    _protectRange = fetchDoubleParam(kParamProtectRange);
    _protectSoft  = fetchDoubleParam(kParamProtectSoft);
    _protectAmt   = fetchDoubleParam(kParamProtectAmt);
    _mode         = fetchChoiceParam(kParamMode);
    _edgeRestore  = fetchDoubleParam(kParamEdgeRestore);
    _spillLight   = fetchChoiceParam(kParamSpillLight);
    _bgGain       = fetchDoubleParam(kParamBGGain);
    _bgBlur       = fetchDoubleParam(kParamBGBlur);
    _wrap         = fetchDoubleParam(kParamWrap);
    _output       = fetchChoiceParam(kParamOutput);
    _alphaOut     = fetchChoiceParam(kParamAlphaOut);
    _invertMask   = fetchBooleanParam(kParamInvertMask);
    _mix          = fetchDoubleParam(kParamMix);
    updateParamsState();
}

void DespillITPlugin::updateParamsState()
{
    _useGPU->setEnabled(cudaIsCompiledIn());
    int ks = 0;
    _keySource->getValue(ks);
    const bool internal = ks == kKeyChoiceInternal || !_keyClip->isConnected();
    _keyLow->setEnabled(internal);
    _keyHigh->setEnabled(internal);
    bool spatial = true;
    _spatial->getValue(spatial);
    _edgeWidth->setEnabled(spatial);
    _edgeFalloff->setEnabled(spatial);
    _interior->setEnabled(spatial);
    const bool bg = _bgClip->isConnected();
    _spillLight->setEnabled(bg);
    _bgGain->setEnabled(bg);
    _bgBlur->setEnabled(bg);
    _wrap->setEnabled(bg);
}

void DespillITPlugin::changedParam(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kParamKeySource || name == kParamSpatial) updateParamsState();
}

void DespillITPlugin::changedClip(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kClipKey || name == kClipBG) updateParamsState();
}

Params DespillITPlugin::paramsAt(double t, double scale)
{
    Params p;
    double r, g, b;
    _screenColor->getValueAtTime(t, r, g, b);
    p.screen[0] = (float)r; p.screen[1] = (float)g; p.screen[2] = (float)b;
    p.screenSource = choiceAt(_screenSource, t);
    if (p.screenSource == kScreenCleanPlate && !_cleanClip->isConnected()) p.screenSource = kScreenFromPlate;
    p.math = choiceAt(_math, t);
    p.bias = (float)_bias->getValueAtTime(t);
    p.limitScale = (float)_limitScale->getValueAtTime(t);
    p.keySource = choiceAt(_keySource, t) == kKeyChoiceAuto && _keyClip->isConnected() ? kKeyInput : kKeyInternal;
    p.keyLow = (float)_keyLow->getValueAtTime(t);
    p.keyHigh = (float)_keyHigh->getValueAtTime(t);
    p.screenBelow = (float)_screenBelow->getValueAtTime(t);
    p.coreAbove = (float)_coreAbove->getValueAtTime(t);
    p.strength = (float)_strength->getValueAtTime(t);
    p.spatial = _spatial->getValueAtTime(t);
    p.edgeWidth = (float)(_edgeWidth->getValueAtTime(t) * scale);
    p.edgeFalloff = (float)(_edgeFalloff->getValueAtTime(t) * scale);
    p.interior = (float)_interior->getValueAtTime(t);
    p.protectHue = (float)_protectHue->getValueAtTime(t);
    p.protectRange = (float)_protectRange->getValueAtTime(t);
    p.protectSoftness = (float)_protectSoft->getValueAtTime(t);
    p.protectAmount = (float)_protectAmt->getValueAtTime(t);
    p.mode = choiceAt(_mode, t);
    p.edgeRestore = (float)_edgeRestore->getValueAtTime(t);
    p.spillLight = choiceAt(_spillLight, t);
    p.bgGain = (float)_bgGain->getValueAtTime(t);
    p.bgBlur = (float)(std::max(0.0, _bgBlur->getValueAtTime(t)) * scale);
    p.wrapAmount = (float)_wrap->getValueAtTime(t);
    p.output = choiceAt(_output, t);
    p.alphaOut = choiceAt(_alphaOut, t);
    p.mix = (float)_mix->getValueAtTime(t);
    return p;
}

bool DespillITPlugin::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if (!_srcClip->isConnected()) return false;
    rod = _srcClip->getRegionOfDefinition(args.time);
    return true;
}

void DespillITPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois)
{
    // the local screen, the distances and the edge colours are computed over the whole
    // rendered window: every input is needed over that window only
    const OfxRectD& r = args.regionOfInterest;
    rois.setRegionOfInterest(*_srcClip, r);
    for (OFX::Clip* c : { _keyClip, _cleanClip, _bgClip, _protectClip, _maskClip })
        if (c->isConnected()) rois.setRegionOfInterest(*c, r);
}

void DespillITPlugin::render(const OFX::RenderArguments& args)
{
    const double t = args.time;
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(t));
    if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (dst->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    const OfxRectI win = dst->getBounds();
    const int w = win.x2 - win.x1, h = win.y2 - win.y1;
    if (w <= 0 || h <= 0) return;

    const double scale = 0.5 * (args.renderScale.x + args.renderScale.y);
    Params p = paramsAt(t, scale);
    p.width = w; p.height = h;

    auto fetch = [&](OFX::Clip* c, bool wanted) -> const OFX::Image* {
        return wanted && c->isConnected() ? c->fetchImage(t) : nullptr;
    };
    std::unique_ptr<const OFX::Image> src(fetch(_srcClip, true));
    std::unique_ptr<const OFX::Image> key(fetch(_keyClip, p.keySource == kKeyInput));
    std::unique_ptr<const OFX::Image> clean(fetch(_cleanClip, p.screenSource == kScreenCleanPlate));
    p.hasBG = _bgClip->isConnected();
    std::unique_ptr<const OFX::Image> bg(fetch(_bgClip, needsBGBlur(p)));
    std::unique_ptr<const OFX::Image> protect(fetch(_protectClip, true));
    std::unique_ptr<const OFX::Image> mask(fetch(_maskClip, true));
    for (const OFX::Image* im : { src.get(), key.get(), clean.get(), bg.get(), protect.get(), mask.get() })
        if (im && im->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    if (!key) p.keySource = kKeyInternal;
    if (!clean && p.screenSource == kScreenCleanPlate) p.screenSource = kScreenFromPlate;
    p.hasBG = bg != nullptr;
    p.hasProtect = protect != nullptr;
    const bool invertMask = _invertMask->getValueAtTime(t);

    // ---- GPU: rows go straight from the images into the transfer memory and back
    if (_useGPU->getValueAtTime(t) && cudaIsCompiledIn()) {
        const RowReader rSrc = [&](int r0, int r1, float* o) { readRows(src.get(), win, r0, r1, false, kRowRGBA, false, o); };
        const RowReader rKey = [&](int r0, int r1, float* o) { readRows(key.get(), win, r0, r1, false, kRowMatte, false, o); };
        const RowReader rClean = [&](int r0, int r1, float* o) { readRows(clean.get(), win, r0, r1, true, kRowRGB, false, o); };
        const RowReader rBG = [&](int r0, int r1, float* o) { readRows(bg.get(), win, r0, r1, true, kRowRGB, false, o); };
        const RowReader rProt = [&](int r0, int r1, float* o) { readRows(protect.get(), win, r0, r1, false, kRowMatte, false, o); };
        const RowReader rMask = [&](int r0, int r1, float* o) { readRows(mask.get(), win, r0, r1, false, kRowMatte, invertMask, o); };
        const RowWriter wOut = [&](int r0, int r1, const float* i) { writeRows(dst.get(), win, r0, r1, i); };
        RowInputs ri;
        ri.src = &rSrc;
        ri.key = key ? &rKey : nullptr;
        ri.clean = clean ? &rClean : nullptr;
        ri.bg = bg ? &rBG : nullptr;
        ri.protect = protect ? &rProt : nullptr;
        ri.mask = mask ? &rMask : nullptr;
        std::string err;
        if (processCUDARows(p, ri, wOut, err)) return;
        OFX::Log::warning(true, "despill_IT_OFX: GPU render failed (%s), falling back to CPU", err.c_str());
    }

    // ---- CPU
    const size_t N = (size_t)w * h;
    std::vector<float> srcBuf(4 * N), keyBuf, cleanBuf, bgBuf, protBuf, maskBuf, outBuf(4 * N);
    auto load = [&](const OFX::Image* im, std::vector<float>& buf, RowMode mode, bool repeat, bool invert) {
        if (!im) return;
        buf.resize((size_t)mode * N);
        parallelRanges(h, [&](int r0, int r1) { readRows(im, win, r0, r1, repeat, mode, invert, &buf[(size_t)mode * r0 * w]); });
    };
    load(src.get(), srcBuf, kRowRGBA, false, false);
    load(key.get(), keyBuf, kRowMatte, false, false);
    load(clean.get(), cleanBuf, kRowRGB, true, false);
    load(bg.get(), bgBuf, kRowRGB, true, false);
    load(protect.get(), protBuf, kRowMatte, false, false);
    load(mask.get(), maskBuf, kRowMatte, false, invertMask);
    Inputs in;
    in.src = srcBuf.data();
    in.key = key ? keyBuf.data() : nullptr;
    in.clean = clean ? cleanBuf.data() : nullptr;
    in.bg = bg ? bgBuf.data() : nullptr;
    in.protect = protect ? protBuf.data() : nullptr;
    in.mask = mask ? maskBuf.data() : nullptr;
    processCPU(p, in, outBuf.data());
    parallelRanges(h, [&](int r0, int r1) { writeRows(dst.get(), win, r0, r1, &outBuf[4 * (size_t)r0 * w]); });
}

// ================================================================= factory
mDeclarePluginFactory(DespillITFactory, {}, ;);

void DespillITFactory::unload() { cudaReleaseResources(); }

void DespillITFactory::describe(OFX::ImageEffectDescriptor& desc)
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

template <class D>
D* finish(D* p, OFX::PageParamDescriptor* page, const char* label, const char* hint)
{
    p->setLabels(label, label, label);
    p->setHint(hint);
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

OFX::ClipDescriptor* optionalClip(OFX::ImageEffectDescriptor& desc, const char* name, bool rgb, bool isMask)
{
    OFX::ClipDescriptor* c = desc.defineClip(name);
    c->addSupportedComponent(OFX::ePixelComponentRGBA);
    if (rgb) c->addSupportedComponent(OFX::ePixelComponentRGB);
    c->addSupportedComponent(OFX::ePixelComponentAlpha);
    c->setOptional(true);
    c->setSupportsTiles(false);
    if (isMask) c->setIsMask(true);
    return c;
}

} // namespace

void DespillITFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    optionalClip(desc, kClipKey, true, false);
    optionalClip(desc, kClipClean, true, false);
    optionalClip(desc, kClipBG, true, false);
    optionalClip(desc, kClipProtect, true, false);
    optionalClip(desc, kClipMask, true, true);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");
    auto divider = [](OFX::ParamDescriptor* p) { p->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintDivider, false); };

    // ---- GPU
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
        page->addChild(*s);
    }

    // ---- screen
    {
        OFX::RGBParamDescriptor* c = desc.defineRGBParam(kParamScreenColor);
        c->setLabels("Screen Color", "Screen Color", "Screen Color");
        c->setHint("Colour of the screen (pick it on a clean, evenly lit part of the screen). Its largest "
                   "component gives the screen channel (green, blue or red). Used as is with Screen Source = "
                   "Global Color, and as the first estimate otherwise.");
        c->setDefault(0.08, 0.45, 0.12);
        c->setDisplayRange(0, 0, 0, 1, 1, 1);
        page->addChild(*c);
    }
    choice(desc, page, kParamScreenSource, "Screen Source",
           "Screen colour each pixel is compared to.\n"
           "Global Color: the Screen Color everywhere.\n"
           "From Plate: the screen is estimated locally from the screen areas of the plate and interpolated "
           "behind the subject (handles uneven screens).\n"
           "Clean Plate: read from the CleanPlate input.",
           { "Global Color", "From Plate", "Clean Plate" }, kScreenFromPlate);
    choice(desc, page, kParamMath, "Despill Math",
           "Limit of the screen channel from the two other channels.\nAverage: weighted average (classic).\n"
           "Max: the larger of the two (gentler on colours close to the screen, e.g. blond hair).",
           { "Average", "Max" }, kMathAverage);
    dbl(desc, page, kParamBias, "Bias",
        "Weight between the two other channels in the limit. Green screen: -1 = red limits green, "
        "+1 = blue limits green, 0 = equal weights.", 0, -1, 1, -1, 1);
    divider(dbl(desc, page, kParamLimitScale, "Limit Scale",
        "Multiplies the limit. Below 1: stronger despill; above 1: lighter despill.", 1, 0, 4, 0.5, 1.5));

    // ---- key
    choice(desc, page, kParamKeySource, "Key Source",
           "Matte that tells the screen from the subject (screen areas, distances, opaque core).\n"
           "Auto: the Key input when it is connected (alpha of a keyer, 1 = subject), otherwise the internal key.\n"
           "Internal: key computed from the amount of screen in each pixel.",
           { "Auto", "Internal" }, kKeyChoiceAuto);
    dbl(desc, page, kParamKeyLow, "Key Low",
        "Internal key: fraction of screen below which a pixel is fully subject.", 0.4, 0, 2, 0, 1);
    dbl(desc, page, kParamKeyHigh, "Key High",
        "Internal key: fraction of screen above which a pixel is fully screen.", 0.9, 0, 2, 0, 1.5);
    dbl(desc, page, kParamScreenBelow, "Screen Below",
        "Key value at or below which a pixel is screen (local screen estimation, distance to the screen).",
        0.05, 0, 1, 0, 0.5);
    divider(dbl(desc, page, kParamCoreAbove, "Core Above",
        "Key value at or above which a pixel is opaque core (its colour is propagated into the edges).",
        0.95, 0, 1, 0.5, 1));

    // ---- strength
    dbl(desc, page, kParamStrength, "Strength", "Overall despill amount.", 1, 0, 4, 0, 1.5);
    boolean(desc, page, kParamSpatial, "Spatial",
            "Full despill near the screen, fading towards the inside of the subject. Off: same despill everywhere.", true);
    dbl(desc, page, kParamEdgeWidth, "Edge Width",
        "Distance from the screen (pixels) over which the despill is full.", 4, 0, 1e4, 0, 50);
    dbl(desc, page, kParamEdgeFalloff, "Edge Falloff",
        "Distance (pixels) over which the despill fades from full to Interior Strength.", 40, 0, 1e4, 0, 200);
    divider(dbl(desc, page, kParamInterior, "Interior Strength",
        "Despill amount far from the screen (reflected spill inside the subject). 0 = none.", 0.15, 0, 1, 0, 1));

    // ---- protection
    dbl(desc, page, kParamProtectHue, "Protect Hue",
        "Hue to protect, in degrees (0 = red, 30 = skin, 60 = yellow, 120 = green, 180 = cyan, 240 = blue).",
        30, 0, 360, 0, 360);
    dbl(desc, page, kParamProtectRange, "Protect Range",
        "Half-width of the protected hue range, in degrees. 0 = no hue protection.", 0, 0, 180, 0, 90);
    dbl(desc, page, kParamProtectSoft, "Protect Softness", "Soft transition around the protected range, in degrees.",
        15, 0, 180, 0, 60);
    divider(dbl(desc, page, kParamProtectAmt, "Protect Amount",
        "Strength of the protection (protected hues and Protect input).", 1, 0, 1, 0, 1));

    // ---- despill
    choice(desc, page, kParamMode, "Despill Mode",
           "How the spill is removed (always relative to the local screen).\n"
           "Screen Channel: the screen channel is limited by the two others (classic despill).\n"
           "Neutralize: the screen colour is turned into neutral light of the same luminance.\n"
           "Subtract Screen: the screen light is removed (physically right for reflected spill; edges get "
           "darker, as in a premultiplied foreground - combine with Spill Light = Background).",
           { "Screen Channel", "Neutralize", "Subtract Screen" }, kModeChannel);
    divider(dbl(desc, page, kParamEdgeRestore, "Edge Restore",
        "Rebuilds the colour of the semi-transparent edges from the opaque core of the subject, keeping their "
        "luminance (avoids grey edges on hair). 0 = off.", 0.75, 0, 1, 0, 1));

    // ---- background
    choice(desc, page, kParamSpillLight, "Spill Light",
           "None: nothing replaces the removed spill.\n"
           "Background: the removed screen light is replaced by the light of the blurred BG input.",
           { "None", "Background" }, kLightNone);
    dbl(desc, page, kParamBGGain, "BG Gain", "Gain of the BG light that replaces the spill.", 1, 0, 100, 0, 4);
    dbl(desc, page, kParamBGBlur, "BG Blur", "Blur of the BG (pixels), for the spill light and the light wrap.",
        20, 0, 1e4, 0, 200);
    divider(dbl(desc, page, kParamWrap, "Light Wrap",
        "Adds the blurred BG on the inner edges of the subject (key x blurred inverse key). 0 = off.", 0, 0, 10, 0, 1));

    // ---- output
    choice(desc, page, kParamOutput, "Output",
           "Result, or a view to set up the effect: spill amount, despill strength (spatial x protection), key, "
           "local screen colour, propagated core colour.",
           { "Result", "Spill Map", "Strength", "Key", "Local Screen", "Core Colour" }, kOutResult);
    choice(desc, page, kParamAlphaOut, "Alpha Output", "Alpha of the result.", { "Source Alpha", "Key", "Spill Map" }, kAlphaSource);
    boolean(desc, page, kParamInvertMask, "Invert Mask", "Inverts the Mask input.", false);
    dbl(desc, page, kParamMix, "Mix", "Effect mix.", 1, 0, 1, 0, 1);
}

OFX::ImageEffect* DespillITFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new DespillITPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static DespillITFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
} // namespace Plugin
} // namespace OFX
