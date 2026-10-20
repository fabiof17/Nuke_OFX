// MasterKeyerPlugin.cpp
// OFX front-end of MasterKeyer_OFX: a complete keyer (matte, colour patches, degrain, spill
// suppression, edge blend, composite) inspired by the Master Keyer of Autodesk Flame. Flame's
// algorithm is not public: this is an independent implementation of the same workflow.
// CPU and CUDA back-ends.
//
// Inputs : Source, BG (optional), Garbage (optional), Holdout (optional)
#include "ofxsImageEffect.h"
#include "ofxKeySyms.h"
#include "OverlayGL.h"
#include "ColourSpaceView.h"
#include "ofxsLog.h"

#include "MasterKeyerCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define kPluginName          "MasterKeyer_OFX"
#define kPluginGrouping      "Keyer"
#define kPluginIdentifier    "fr.masterkeyer.MasterKeyer_OFX"
#define kPluginVersionMajor  1
#define kPluginVersionMinor  0

#define kPluginDescription \
    "MasterKeyer_OFX: complete keyer inspired by the Master Keyer of Autodesk Flame.\n\n" \
    "Pick the screen with the Primary colour, then refine: Secondary colour and Mix for the transparency " \
    "level, gain, clips and softness of the matte, colour patches, degrain of the plate used for the matte, " \
    "automatic spill suppression fading towards the inside of the subject, and edge blend (luma, colour, " \
    "background colour) before the composite over the BG input.\n\n" \
    "Flame's algorithm is not public: this is an independent implementation of the same workflow, not a " \
    "copy, and results differ from Flame.\n" \
    "Optional CUDA GPU acceleration (NVIDIA cards), automatic fallback to the CPU."

static const char* kClipBG      = "BG";
static const char* kClipKeyIn   = "KeyIn";
static const char* kClipGarbage = "Garbage";
static const char* kClipHoldout = "Holdout";

static const char* kParamUseGPU       = "useGPU";
static const char* kParamGPUStatus    = "gpuStatus";
static const char* kParamPrimary      = "primary";
static const char* kParamSecondary    = "secondary";
static const char* kParamMix          = "secondaryMix";
static const char* kParamScreenSource = "screenSource";
static const char* kParamBalance      = "balance";
static const char* kParamMatteMode    = "matteMode";
static const char* kParamTolerance    = "tolerance";
static const char* kParamSoftness3D   = "softness3D";
static const char* kParamLumaWeight   = "lumaWeight";
static const char* kParamShow3D       = "show3DView";
static const char* kParamViewCorner   = "viewCorner";
static const char* kParamViewYaw      = "viewYaw";       // hidden: orientation and zoom of the 3D view
static const char* kParamViewPitch    = "viewPitch";
static const char* kParamViewZoom     = "viewZoom";
static const char* kParamViewSize     = "viewSize";
static const char* kParamTolDisplay   = "toleranceDisplay";
static const char* kParamEnvDisplay   = "softnessDisplay";
static const char* kParamViewSpace    = "viewSpace";      // hidden: size of the 3D view (screen pixels)
static const char* kParamGain         = "gain";
static const char* kParamShadows      = "shadows";
static const char* kParamMidtones     = "midtones";
static const char* kParamHighlights   = "highlights";
static const char* kParamResetMatte   = "resetMatte";
static const char* kParamResetSpill   = "resetSpill";
static const char* kParamResetBlend   = "resetBlend";
static const char* kParamResetBalance = "resetEdgeBalance";
static const char* kParamResetDegrain = "resetDegrain";
static const char* kParamSpillColour  = "spillColour";
static const char* kParamPickSpill    = "pickSpillColour";
static const char* kParamPickGrain    = "pickGrain";
static const char* kParamKeyPicked    = "primaryPicked";
static const char* kParamUndoPick     = "undoPick";       // Matte tab
static const char* kParamUndoPick2    = "undoPickPatches"; // same action, Patches tab
static const char* kParamPickHistory  = "pickHistory";    // hidden: values before each pick  // hidden: the Primary has been picked once
static const char* kParamClipLow      = "clipBlack";
static const char* kParamClipHigh     = "clipWhite";
static const char* kParamSoftness     = "softness";
static const char* kParamShrink       = "shrinkGrow";
static const char* kParamMatteBlur    = "matteBlur";
static const char* kParamOutput       = "output";
static const char* kParamDegrain      = "degrain";
static const char* kParamGrainSize    = "grainSize";
static const char* kParamGrainEdges   = "grainEdges";
static const char* kParamGrainMore    = "grainMore";
static const char* kParamAutoCC       = "autoCC";
static const char* kParamSpillStr     = "spillStrength";
static const char* kParamSpillRange   = "spillRange";
static const char* kParamSpillInt     = "spillInterior";
static const char* kParamHue          = "spillHue";
static const char* kParamEdgeSize     = "edgeSize";
static const char* kParamEdgeLuma     = "edgeLuma";
static const char* kParamEdgeBalance  = "edgeBalance";
static const char* kParamEdgeFromBG   = "edgeFromBG";
static const char* kParamBGBlur       = "bgBlur";
static const char* kParamPickPrimary  = "pickPrimary";
static const char* kParamPickSecond   = "pickSecondary";
static const char* kParamPickTarget   = "pickTarget";     // see PickTarget
static const char* kParamPickA        = "pickCorner1";    // picked area (canonical coordinates)
static const char* kParamPickB        = "pickCorner2";
static const char* kParamPickTrigger  = "pickTrigger";    // bumped by the overlay when a pick is done

static const char* kPropLayoutHint = "OfxParamPropLayoutHint";
static const int   kLayoutHintNoNewLine = 2;
static const int   kLayoutHintDivider   = 1;

using namespace mk;

// What the viewer picker samples.
enum PickTarget {
    kPickNone = 0, kPickPrimary = 1, kPickSecondary = 2,
    kPickPatch = 3,                    // + k: replace the colours of patch k
    kPickPatchAdd = 3 + kPatches,      // + k: add colours to patch k
    kPickSpill = 3 + 2 * kPatches,
    kPickGrain = 4 + 2 * kPatches
};
enum PatchChoice { kChoiceAuto = 0 };  // Type menu: Auto, Black, White, Edge Analysis (= PatchType values)

static std::string patchName(int k, const char* what) { return "patch" + std::to_string(k + 1) + what; }

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
// RGBA (RGB images get alpha 1), RGB, or a matte (alpha; luminance for RGB images; clamped
// to [0, 1]). Outside the image: black, or (repeatEdges) the edge pixels repeated.
void readRows(const OFX::Image* img, const OfxRectI& rect, int r0, int r1, bool repeatEdges, RowMode mode, float* out)
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
            const float m = nc == 3 ? 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2] : c[3];
            o[0] = std::min(1.f, std::max(0.f, m));
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

// Back to the default value, animation removed.
inline void resetParam(OFX::DoubleParam* p) { p->deleteAllKeys(); p->setValue(p->getDefault()); }
inline void resetParam(OFX::IntParam* p) { p->deleteAllKeys(); p->setValue(p->getDefault()); }
inline void resetParam(OFX::BooleanParam* p) { bool v; p->getDefault(v); p->deleteAllKeys(); p->setValue(v); }
inline void resetParam(OFX::ChoiceParam* p) { int v; p->getDefault(v); p->deleteAllKeys(); p->setValue(v); }
inline void resetParam(OFX::Double3DParam* p)
{
    double x, y, z;
    p->getDefault(x, y, z);
    p->deleteAllKeys();
    p->setValue(x, y, z);
}
inline void resetParam(OFX::RGBParam* p)
{
    double r, g, b;
    p->getDefault(r, g, b);
    p->deleteAllKeys();
    p->setValue(r, g, b);
}

inline void rgbAt(OFX::Double3DParam* p, double t, float* o)
{
    double r, g, b;
    p->getValueAtTime(t, r, g, b);
    o[0] = (float)r; o[1] = (float)g; o[2] = (float)b;
}

inline void rgbAt(OFX::RGBParam* p, double t, float* o)
{
    double r, g, b;
    p->getValueAtTime(t, r, g, b);
    o[0] = (float)r; o[1] = (float)g; o[2] = (float)b;
}

} // namespace

// ================================================================== plugin
class MasterKeyerPlugin : public OFX::ImageEffect
{
public:
    explicit MasterKeyerPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;
    void changedClip(const OFX::InstanceChangedArgs& args, const std::string& clipName) override;

private:
    Params paramsAt(double t, double renderScale);
    void updateParamsState();
    void pickFromSource(double t);
    void savePickState(double t);
    void undoPick();
    void estimateGrain(const std::vector<float>& rgb, int w, int h, double t);
    void resetMatte();
    void resetSpill();
    void resetBlend();
    void resetDegrain();

    OFX::Clip *_dstClip, *_srcClip, *_keyInClip, *_bgClip, *_garbageClip, *_holdoutClip;

    OFX::BooleanParam* _useGPU;
    OFX::Double3DParam* _primary;
    OFX::Double3DParam* _secondary;
    OFX::DoubleParam*  _mix;
    OFX::ChoiceParam*  _screenSource;
    OFX::DoubleParam*  _balance;
    OFX::ChoiceParam*  _matteMode;
    OFX::DoubleParam*  _tolerance;
    OFX::DoubleParam*  _softness3D;
    OFX::DoubleParam*  _lumaWeight;
    OFX::DoubleParam*  _gain;
    OFX::DoubleParam*  _shadows;
    OFX::DoubleParam*  _midtones;
    OFX::DoubleParam*  _highlights;
    OFX::DoubleParam*  _clipLow;
    OFX::DoubleParam*  _clipHigh;
    OFX::DoubleParam*  _softness;
    OFX::IntParam*     _shrink;
    OFX::DoubleParam*  _matteBlur;
    OFX::ChoiceParam*  _output;
    OFX::BooleanParam* _patchOn[kPatches];
    OFX::ChoiceParam*  _patchType[kPatches];
    OFX::Double3DParam* _patchLo[kPatches];
    OFX::Double3DParam* _patchHi[kPatches];
    OFX::DoubleParam*  _patchRange[kPatches];
    OFX::DoubleParam*  _patchSoft[kPatches];
    OFX::BooleanParam* _degrain;
    OFX::DoubleParam*  _grainSize;
    OFX::DoubleParam*  _grainEdges;
    OFX::BooleanParam* _grainMore;
    OFX::BooleanParam* _autoCC;
    OFX::DoubleParam*  _spillStr;
    OFX::DoubleParam*  _spillRange;
    OFX::DoubleParam*  _spillInt;
    OFX::DoubleParam*  _hue;
    OFX::Double3DParam* _spillColour;
    OFX::DoubleParam*  _edgeSize;
    OFX::DoubleParam*  _edgeLuma;
    OFX::RGBParam*     _edgeBalance;
    OFX::DoubleParam*  _edgeFromBG;
    OFX::DoubleParam*  _bgBlur;
};

MasterKeyerPlugin::MasterKeyerPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _dstClip     = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip     = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _keyInClip   = fetchClip(kClipKeyIn);
    _bgClip      = fetchClip(kClipBG);
    _garbageClip = fetchClip(kClipGarbage);
    _holdoutClip = fetchClip(kClipHoldout);

    _useGPU       = fetchBooleanParam(kParamUseGPU);
    _primary      = fetchDouble3DParam(kParamPrimary);
    _secondary    = fetchDouble3DParam(kParamSecondary);
    _mix          = fetchDoubleParam(kParamMix);
    _screenSource = fetchChoiceParam(kParamScreenSource);
    _balance      = fetchDoubleParam(kParamBalance);
    _matteMode    = fetchChoiceParam(kParamMatteMode);
    _tolerance    = fetchDoubleParam(kParamTolerance);
    _softness3D   = fetchDoubleParam(kParamSoftness3D);
    _lumaWeight   = fetchDoubleParam(kParamLumaWeight);
    _gain         = fetchDoubleParam(kParamGain);
    _shadows      = fetchDoubleParam(kParamShadows);
    _midtones     = fetchDoubleParam(kParamMidtones);
    _highlights   = fetchDoubleParam(kParamHighlights);
    _clipLow      = fetchDoubleParam(kParamClipLow);
    _clipHigh     = fetchDoubleParam(kParamClipHigh);
    _softness     = fetchDoubleParam(kParamSoftness);
    _shrink       = fetchIntParam(kParamShrink);
    _matteBlur    = fetchDoubleParam(kParamMatteBlur);
    _output       = fetchChoiceParam(kParamOutput);
    for (int k = 0; k < kPatches; ++k) {
        _patchOn[k]    = fetchBooleanParam(patchName(k, "Enable"));
        _patchType[k]  = fetchChoiceParam(patchName(k, "Type"));
        _patchLo[k]    = fetchDouble3DParam(patchName(k, "Low"));
        _patchHi[k]    = fetchDouble3DParam(patchName(k, "High"));
        _patchRange[k] = fetchDoubleParam(patchName(k, "Range"));
        _patchSoft[k]  = fetchDoubleParam(patchName(k, "Softness"));
    }
    _degrain      = fetchBooleanParam(kParamDegrain);
    _grainSize    = fetchDoubleParam(kParamGrainSize);
    _grainEdges   = fetchDoubleParam(kParamGrainEdges);
    _grainMore    = fetchBooleanParam(kParamGrainMore);
    _autoCC       = fetchBooleanParam(kParamAutoCC);
    _spillStr     = fetchDoubleParam(kParamSpillStr);
    _spillRange   = fetchDoubleParam(kParamSpillRange);
    _spillInt     = fetchDoubleParam(kParamSpillInt);
    _hue          = fetchDoubleParam(kParamHue);
    _spillColour  = fetchDouble3DParam(kParamSpillColour);
    _edgeSize     = fetchDoubleParam(kParamEdgeSize);
    _edgeLuma     = fetchDoubleParam(kParamEdgeLuma);
    _edgeBalance  = fetchRGBParam(kParamEdgeBalance);
    _edgeFromBG   = fetchDoubleParam(kParamEdgeFromBG);
    _bgBlur       = fetchDoubleParam(kParamBGBlur);
    updateParamsState();
}

void MasterKeyerPlugin::updateParamsState()
{
    _useGPU->setEnabled(cudaIsCompiledIn());
    int mode = 0;
    _matteMode->getValue(mode);
    _balance->setEnabled(mode != kMatte3D);
    for (OFX::DoubleParam* d : { _tolerance, _softness3D, _lumaWeight }) d->setEnabled(mode != kMatteDifference);
    fetchBooleanParam(kParamShow3D)->setEnabled(mode != kMatteDifference);
    fetchChoiceParam(kParamViewCorner)->setEnabled(mode != kMatteDifference);
    fetchChoiceParam(kParamTolDisplay)->setEnabled(mode != kMatteDifference);
    fetchChoiceParam(kParamEnvDisplay)->setEnabled(mode != kMatteDifference);
    fetchChoiceParam(kParamViewSpace)->setEnabled(mode != kMatteDifference);
    bool on = false;
    _degrain->getValue(on);
    _grainSize->setEnabled(on);
    _grainEdges->setEnabled(on);
    _grainMore->setEnabled(on);
    _autoCC->getValue(on);
    _spillStr->setEnabled(on);
    _spillRange->setEnabled(on);
    _spillInt->setEnabled(on);
    _hue->setEnabled(on);
    _spillColour->setEnabled(on);
    for (int k = 0; k < kPatches; ++k) {
        bool en = false;
        _patchOn[k]->getValue(en);
        for (OFX::Param* q : std::initializer_list<OFX::Param*>{ _patchType[k], _patchLo[k], _patchHi[k], _patchRange[k],
                                                                 _patchSoft[k] })
            q->setEnabled(en);
    }
    const bool bg = _bgClip->isConnected();
    _edgeFromBG->setEnabled(bg);
    _bgBlur->setEnabled(bg);
}

void MasterKeyerPlugin::changedParam(const OFX::InstanceChangedArgs& args, const std::string& name)
{
    // "Pick" buttons: arm (or disarm) the viewer picker for that colour
    int target = kPickNone;
    if (name == kParamPickPrimary) target = kPickPrimary;
    else if (name == kParamPickSecond) target = kPickSecondary;
    else if (name == kParamPickSpill) target = kPickSpill;
    else if (name == kParamPickGrain) target = kPickGrain;
    for (int k = 0; k < kPatches; ++k) {
        if (name == patchName(k, "Pick")) target = kPickPatch + k;
        if (name == patchName(k, "Add")) target = kPickPatchAdd + k;
    }
    if (target) {
        OFX::IntParam* tp = fetchIntParam(kParamPickTarget);
        tp->setValue(tp->getValue() == target ? 0 : target);
        return;
    }
    for (int k = 0; k < kPatches; ++k)
        if (name == patchName(k, "Reset")) {
            resetParam(_patchOn[k]);
            resetParam(_patchType[k]);
            resetParam(_patchLo[k]);
            resetParam(_patchHi[k]);
            resetParam(_patchRange[k]);
            resetParam(_patchSoft[k]);
            fetchIntParam(kParamPickTarget)->setValue(0);
            return;
        }
    if (name == kParamResetMatte) { resetMatte(); return; }
    if (name == kParamResetSpill) { resetSpill(); return; }
    if (name == kParamResetBlend) { resetBlend(); return; }
    if (name == kParamResetBalance) { resetParam(_edgeBalance); return; }
    if (name == kParamResetDegrain) { resetDegrain(); return; }
    if (name == kParamUndoPick || name == kParamUndoPick2) { undoPick(); return; }
    if (name == kParamPickTrigger && args.reason != OFX::eChangeTime) {
        pickFromSource(args.time);
        return;
    }
    if (name == kParamDegrain || name == kParamAutoCC || name == kParamMatteMode || name.find("Enable") != std::string::npos) updateParamsState();
}

// Section resets (Flame: Reset next to the sampling box). Mix and the key colours are kept.
void MasterKeyerPlugin::resetMatte()
{
    for (OFX::DoubleParam* d : { _balance, _tolerance, _softness3D, _lumaWeight, _gain, _shadows, _midtones, _highlights, _clipLow, _clipHigh, _softness,
                                 _matteBlur })
        resetParam(d);
    resetParam(_shrink);
    resetParam(_screenSource);
}

void MasterKeyerPlugin::resetSpill()
{
    for (OFX::DoubleParam* d : { _spillStr, _spillRange, _spillInt, _hue }) resetParam(d);
    double r, g, b;
    _primary->getValue(r, g, b);
    _spillColour->deleteAllKeys();
    _spillColour->setValue(r, g, b);
}

void MasterKeyerPlugin::resetBlend()
{
    for (OFX::DoubleParam* d : { _edgeSize, _edgeLuma, _edgeFromBG, _bgBlur }) resetParam(d);
}

void MasterKeyerPlugin::resetDegrain()
{
    resetParam(_degrain);
    resetParam(_grainSize);
    resetParam(_grainEdges);
    resetParam(_grainMore);
    updateParamsState();
}

namespace {
template <class P, class... V>
void setNow(P* p, double t, V... v)
{
    if (p->getNumKeys() > 0) p->setValueAtTime(t, v...);
    else p->setValue(v...);
}
} // namespace

// Undo Pick. Nuke does not record in its undo history the parameters an effect sets while it
// handles a change (the picks are applied there, after reading the image), so the effect keeps
// its own history: the values of every parameter a pick can change, saved before each pick.
namespace {
const char* kHistSep = "\n#\n";
const size_t kHistMax = 20;
} // namespace

void MasterKeyerPlugin::savePickState(double t)
{
    std::string st;
    char buf[256];
    auto d3 = [&](OFX::Double3DParam* p) {
        double a, b, c;
        p->getValueAtTime(t, a, b, c);
        std::snprintf(buf, sizeof buf, "3 %s %.9g %.9g %.9g %.9g\n", p->getName().c_str(), t, a, b, c);
        st += buf;
    };
    auto d1 = [&](OFX::DoubleParam* p) {
        std::snprintf(buf, sizeof buf, "1 %s %.9g %.9g\n", p->getName().c_str(), t, p->getValueAtTime(t));
        st += buf;
    };
    auto b1 = [&](OFX::BooleanParam* p) {
        std::snprintf(buf, sizeof buf, "b %s %d\n", p->getName().c_str(), p->getValue() ? 1 : 0);
        st += buf;
    };
    auto c1 = [&](OFX::ChoiceParam* p) {
        int v = 0;
        p->getValue(v);
        std::snprintf(buf, sizeof buf, "c %s %d\n", p->getName().c_str(), v);
        st += buf;
    };
    d3(_primary); d3(_secondary); d3(_spillColour);
    d1(_grainSize);
    b1(_autoCC); b1(_degrain); b1(fetchBooleanParam(kParamKeyPicked));
    for (int k = 0; k < kPatches; ++k) { b1(_patchOn[k]); c1(_patchType[k]); d3(_patchLo[k]); d3(_patchHi[k]); }

    OFX::StringParam* h = fetchStringParam(kParamPickHistory);
    std::string all;
    h->getValue(all);
    all = all.empty() ? st : all + kHistSep + st;
    // keep the last kHistMax states
    size_t count = 1, pos = 0;
    while ((pos = all.find(kHistSep, pos)) != std::string::npos) { ++count; pos += 3; }
    while (count > kHistMax) {
        const size_t cut = all.find(kHistSep);
        all.erase(0, cut + 3);
        --count;
    }
    h->setValue(all);
}

void MasterKeyerPlugin::undoPick()
{
    OFX::StringParam* h = fetchStringParam(kParamPickHistory);
    std::string all;
    h->getValue(all);
    if (all.empty()) return;
    const size_t cut = all.rfind(kHistSep);
    const std::string st = cut == std::string::npos ? all : all.substr(cut + 3);
    h->setValue(cut == std::string::npos ? std::string() : all.substr(0, cut));
    size_t start = 0;
    while (start < st.size()) {
        size_t end = st.find('\n', start);
        if (end == std::string::npos) end = st.size();
        const std::string line = st.substr(start, end - start);
        start = end + 1;
        char kind = 0, name[128] = { 0 };
        double t = 0, a = 0, b = 0, c = 0;
        int v = 0;
        if (line.size() < 3) continue;
        kind = line[0];
        if (kind == '3' && std::sscanf(line.c_str() + 2, "%127s %lg %lg %lg %lg", name, &t, &a, &b, &c) == 5) {
            OFX::Double3DParam* p = fetchDouble3DParam(name);
            if (p->getNumKeys() > 0) p->setValueAtTime(t, a, b, c); else p->setValue(a, b, c);
        } else if (kind == '1' && std::sscanf(line.c_str() + 2, "%127s %lg %lg", name, &t, &a) == 3) {
            OFX::DoubleParam* p = fetchDoubleParam(name);
            if (p->getNumKeys() > 0) p->setValueAtTime(t, a); else p->setValue(a);
        } else if (kind == 'b' && std::sscanf(line.c_str() + 2, "%127s %d", name, &v) == 2) {
            fetchBooleanParam(name)->setValue(v != 0);
        } else if (kind == 'c' && std::sscanf(line.c_str() + 2, "%127s %d", name, &v) == 2) {
            fetchChoiceParam(name)->setValue(v);
        }
    }
    updateParamsState();
}

// The picked area of the Key In input (or the Source; the Source for the Spill Colour) - never
// the keyed output - is read here, then used according to the armed target.
void MasterKeyerPlugin::pickFromSource(double t)
{
    OFX::IntParam* tp = fetchIntParam(kParamPickTarget);
    const int target = tp->getValue();
    tp->setValue(kPickNone);
    if (target == kPickNone) return;
    OFX::Clip* clip = target != kPickSpill && _keyInClip->isConnected() ? _keyInClip : _srcClip;
    if (!clip->isConnected()) return;
    std::unique_ptr<const OFX::Image> img(clip->fetchImage(t));
    if (!img || img->getPixelDepth() != OFX::eBitDepthFloat) return;
    const int nc = componentCount(img->getPixelComponents());
    if (nc < 3) return;
    double ax, ay, bx, by;
    fetchDouble2DParam(kParamPickA)->getValue(ax, ay);
    fetchDouble2DParam(kParamPickB)->getValue(bx, by);
    double par = clip->getPixelAspectRatio();
    if (!(par > 0.0)) par = 1.0;
    const OfxPointD sc = img->getRenderScale();
    const OfxRectI bd = img->getBounds();
    // canonical -> pixel; a click (empty area) takes the pixel under the cursor
    int x1 = (int)std::floor(std::min(ax, bx) * sc.x / par), x2 = (int)std::ceil(std::max(ax, bx) * sc.x / par);
    int y1 = (int)std::floor(std::min(ay, by) * sc.y), y2 = (int)std::ceil(std::max(ay, by) * sc.y);
    if (x2 <= x1) x2 = x1 + 1;
    if (y2 <= y1) y2 = y1 + 1;
    x1 = std::max(x1, bd.x1); x2 = std::min(x2, bd.x2);
    y1 = std::max(y1, bd.y1); y2 = std::min(y2, bd.y2);
    if (x1 >= x2 || y1 >= y2) return;
    const int aw = x2 - x1, ah = y2 - y1;
    std::vector<float> rgb((size_t)3 * aw * ah);
    double sum[3] = { 0, 0, 0 }, sq[3] = { 0, 0, 0 };
    float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
    long long n = 0;
    for (int y = y1; y < y2; ++y) {
        const float* row = static_cast<const float*>(img->getPixelAddress(x1, y));
        for (int x = x1; x < x2; ++x) {
            float* o = &rgb[3 * ((size_t)(y - y1) * aw + (x - x1))];
            const float* px = row ? row + (size_t)(x - x1) * nc : nullptr;
            for (int c = 0; c < 3; ++c) o[c] = px && std::isfinite(px[c]) ? px[c] : 0.f;
            if (!px) continue;
            for (int c = 0; c < 3; ++c) {
                sum[c] += o[c]; sq[c] += (double)o[c] * o[c];
                mn[c] = std::min(mn[c], o[c]); mx[c] = std::max(mx[c], o[c]);
            }
            ++n;
        }
    }
    if (n == 0) return;
    savePickState(t);                                  // for Undo Pick
    double mean[3], sd[3];
    for (int c = 0; c < 3; ++c) {
        mean[c] = sum[c] / n;
        sd[c] = std::sqrt(std::max(0.0, sq[c] / n - mean[c] * mean[c]));
    }

    switch (target) {
    case kPickPrimary: {
        double pr, pg, pb, sr, sg, sb;
        _primary->getValueAtTime(t, pr, pg, pb);
        _spillColour->getValueAtTime(t, sr, sg, sb);
        // the spill colour follows the Primary as long as it has not been changed
        const bool spillFollows = std::fabs(sr - pr) < 1e-6 && std::fabs(sg - pg) < 1e-6 && std::fabs(sb - pb) < 1e-6;
        setNow(_primary, t, mean[0], mean[1], mean[2]);
        if (spillFollows) setNow(_spillColour, t, mean[0], mean[1], mean[2]);
        OFX::BooleanParam* picked = fetchBooleanParam(kParamKeyPicked);
        if (!picked->getValue()) {                     // first key colour: the Secondary too,
            setNow(_secondary, t, mean[0], mean[1], mean[2]);
            picked->setValue(true);
            _autoCC->setValue(true);                   // and auto colour suppression kicks in
            updateParamsState();
        }
        break;
    }
    case kPickSecondary:
        setNow(_secondary, t, mean[0], mean[1], mean[2]);
        break;
    case kPickSpill:
        setNow(_spillColour, t, mean[0], mean[1], mean[2]);
        break;
    case kPickGrain:
        estimateGrain(rgb, aw, ah, t);
        break;
    default: {
        const bool add = target >= kPickPatchAdd;
        const int k = add ? target - kPickPatchAdd : target - kPickPatch;
        if (k < 0 || k >= kPatches) break;
        // colour box of the sample: mean +- 1.5 sigma, within the sampled values (robust to noise)
        double lo[3], hi[3];
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::max<double>(mn[c], mean[c] - 1.5 * sd[c]);
            hi[c] = std::min<double>(mx[c], mean[c] + 1.5 * sd[c]);
        }
        bool on = false;
        _patchOn[k]->getValue(on);
        if (add && on) {
            double l[3], h[3];
            _patchLo[k]->getValueAtTime(t, l[0], l[1], l[2]);
            _patchHi[k]->getValueAtTime(t, h[0], h[1], h[2]);
            for (int c = 0; c < 3; ++c) { lo[c] = std::min(lo[c], l[c]); hi[c] = std::max(hi[c], h[c]); }
        }
        // automatic type: Black where the matte is mostly transparent, White where it is mostly opaque
        if (choiceAt(_patchType[k], t) == kChoiceAuto) {
            const Params p = paramsAt(t, 1.0);
            const float c3[3] = { (float)mean[0], (float)mean[1], (float)mean[2] };
            const float a = rawMatte(p, c3, p.primary);
            _patchType[k]->setValue(a < 0.5f ? (int)kPatchBlack : (int)kPatchWhite);
        }
        setNow(_patchLo[k], t, lo[0], lo[1], lo[2]);
        setNow(_patchHi[k], t, hi[0], hi[1], hi[2]);
        _patchOn[k]->setValue(true);
        updateParamsState();
        break;
    }
    }
}

// Pick Grain: Size measured on the sampled area, Degrain on.
void MasterKeyerPlugin::estimateGrain(const std::vector<float>& rgb, int w, int h, double t)
{
    const double size = estimateGrainSize(rgb.data(), w, h);
    if (size <= 0.0) return;
    setNow(_grainSize, t, size);
    _degrain->setValue(true);
    updateParamsState();
}

void MasterKeyerPlugin::changedClip(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kClipBG) updateParamsState();
}

Params MasterKeyerPlugin::paramsAt(double t, double scale)
{
    Params p;
    rgbAt(_primary, t, p.primary);
    rgbAt(_secondary, t, p.secondary);
    p.mix = (float)_mix->getValueAtTime(t);
    p.screenSource = choiceAt(_screenSource, t);
    p.balance = (float)_balance->getValueAtTime(t);
    p.matteMode = choiceAt(_matteMode, t);
    p.tolerance = (float)_tolerance->getValueAtTime(t);
    p.softness3D = (float)_softness3D->getValueAtTime(t);
    p.lumaWeight = (float)_lumaWeight->getValueAtTime(t);
    p.gain = (float)_gain->getValueAtTime(t);
    p.shadows = (float)_shadows->getValueAtTime(t);
    p.midtones = (float)_midtones->getValueAtTime(t);
    p.highlights = (float)_highlights->getValueAtTime(t);
    p.clipLow = (float)_clipLow->getValueAtTime(t);
    p.clipHigh = (float)_clipHigh->getValueAtTime(t);
    p.softness = (float)_softness->getValueAtTime(t);
    p.shrink = (int)std::lround(_shrink->getValueAtTime(t) * scale);
    p.matteBlur = (float)(std::max(0.0, _matteBlur->getValueAtTime(t)) * scale);
    p.output = choiceAt(_output, t);
    for (int k = 0; k < kPatches; ++k) {
        p.patch[k].type = _patchOn[k]->getValueAtTime(t) ? choiceAt(_patchType[k], t) : (int)kPatchOff;
        rgbAt(_patchLo[k], t, p.patch[k].lo);
        rgbAt(_patchHi[k], t, p.patch[k].hi);
        for (int c = 0; c < 3; ++c)                    // fields edited by hand in the wrong order
            if (p.patch[k].lo[c] > p.patch[k].hi[c]) std::swap(p.patch[k].lo[c], p.patch[k].hi[c]);
        p.patch[k].range = (float)_patchRange[k]->getValueAtTime(t);
        p.patch[k].soft = (float)_patchSoft[k]->getValueAtTime(t);
    }
    p.degrain = _degrain->getValueAtTime(t);
    p.grainSize = (float)(std::max(0.0, _grainSize->getValueAtTime(t)) * scale);
    p.grainEdges = (float)_grainEdges->getValueAtTime(t);
    p.grainMore = _grainMore->getValueAtTime(t);
    p.autoCC = _autoCC->getValueAtTime(t);
    p.spillStrength = (float)_spillStr->getValueAtTime(t);
    p.spillRange = (float)(std::max(0.0, _spillRange->getValueAtTime(t)) * scale);
    p.spillInterior = (float)_spillInt->getValueAtTime(t);
    p.hue = (float)_hue->getValueAtTime(t);
    rgbAt(_spillColour, t, p.spillColour);
    p.edgeSize = (float)(std::max(0.0, _edgeSize->getValueAtTime(t)) * scale);
    p.edgeLuma = (float)_edgeLuma->getValueAtTime(t);
    rgbAt(_edgeBalance, t, p.edgeBalance);
    p.edgeFromBG = (float)_edgeFromBG->getValueAtTime(t);
    p.bgBlur = (float)(std::max(0.0, _bgBlur->getValueAtTime(t)) * scale);
    return p;
}

bool MasterKeyerPlugin::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if (!_srcClip->isConnected()) return false;
    rod = _srcClip->getRegionOfDefinition(args.time);
    return true;
}

void MasterKeyerPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois)
{
    // all the spatial stages work on the rendered window: every input is needed over it only
    const OfxRectD& r = args.regionOfInterest;
    rois.setRegionOfInterest(*_srcClip, r);
    for (OFX::Clip* c : { _keyInClip, _bgClip, _garbageClip, _holdoutClip })
        if (c->isConnected()) rois.setRegionOfInterest(*c, r);
}

void MasterKeyerPlugin::render(const OFX::RenderArguments& args)
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

    auto fetch = [&](OFX::Clip* c) -> const OFX::Image* { return c->isConnected() ? c->fetchImage(t) : nullptr; };
    std::unique_ptr<const OFX::Image> src(fetch(_srcClip));
    std::unique_ptr<const OFX::Image> keyIn(fetch(_keyInClip));
    std::unique_ptr<const OFX::Image> bg(fetch(_bgClip));
    std::unique_ptr<const OFX::Image> garbage(fetch(_garbageClip));
    std::unique_ptr<const OFX::Image> holdout(fetch(_holdoutClip));
    for (const OFX::Image* im : { src.get(), keyIn.get(), bg.get(), garbage.get(), holdout.get() })
        if (im && im->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    p.hasBG = bg != nullptr;
    p.hasGarbage = garbage != nullptr;
    p.hasHoldout = holdout != nullptr;

    // ---- GPU: rows go straight from the images into the transfer memory and back
    if (_useGPU->getValueAtTime(t) && cudaIsCompiledIn()) {
        const RowReader rSrc = [&](int r0, int r1, float* o) { readRows(src.get(), win, r0, r1, false, kRowRGB, o); };
        const RowReader rKey = [&](int r0, int r1, float* o) { readRows(keyIn.get(), win, r0, r1, false, kRowRGB, o); };
        const RowReader rBG = [&](int r0, int r1, float* o) { readRows(bg.get(), win, r0, r1, true, kRowRGB, o); };
        const RowReader rGb = [&](int r0, int r1, float* o) { readRows(garbage.get(), win, r0, r1, false, kRowMatte, o); };
        const RowReader rHo = [&](int r0, int r1, float* o) { readRows(holdout.get(), win, r0, r1, false, kRowMatte, o); };
        const RowWriter wOut = [&](int r0, int r1, const float* i) { writeRows(dst.get(), win, r0, r1, i); };
        RowInputs ri;
        ri.src = &rSrc;
        ri.bg = bg ? &rBG : nullptr;
        ri.keyIn = keyIn ? &rKey : nullptr;
        ri.garbage = garbage ? &rGb : nullptr;
        ri.holdout = holdout ? &rHo : nullptr;
        std::string err;
        if (processCUDARows(p, ri, wOut, err)) return;
        OFX::Log::warning(true, "MasterKeyer_OFX: GPU render failed (%s), falling back to CPU", err.c_str());
    }

    // ---- CPU
    const size_t N = (size_t)w * h;
    std::vector<float> srcBuf(4 * N), keyBuf, bgBuf, gbBuf, hoBuf, outBuf(4 * N);
    auto load = [&](const OFX::Image* im, std::vector<float>& buf, RowMode mode, bool repeat) {
        if (!im) return;
        buf.resize((size_t)mode * N);
        parallelRanges(h, [&](int r0, int r1) { readRows(im, win, r0, r1, repeat, mode, &buf[(size_t)mode * r0 * w]); });
    };
    load(src.get(), srcBuf, kRowRGBA, false);
    load(keyIn.get(), keyBuf, kRowRGB, false);
    load(bg.get(), bgBuf, kRowRGB, true);
    load(garbage.get(), gbBuf, kRowMatte, false);
    load(holdout.get(), hoBuf, kRowMatte, false);
    Inputs in;
    in.src = srcBuf.data();
    in.bg = bg ? bgBuf.data() : nullptr;
    in.keyIn = keyIn ? keyBuf.data() : nullptr;
    in.garbage = garbage ? gbBuf.data() : nullptr;
    in.holdout = holdout ? hoBuf.data() : nullptr;
    processCPU(p, in, outBuf.data());
    parallelRanges(h, [&](int r0, int r1) { writeRows(dst.get(), win, r0, r1, &outBuf[4 * (size_t)r0 * w]); });
}

// ================================================================= factory
// ================================================================== viewer overlay
// Colour picker that reads the Source: after a "Pick" button, click in the viewer (one pixel)
// or drag (average of an area). Nuke's own colour picker reads the output of the node, which
// is keyed and premultiplied. Esc or the same button again cancels.
namespace {

class PickOverlay : public OFX::OverlayInteract
{
public:
    PickOverlay(OfxInteractHandle handle, OFX::ImageEffect* effect) : OFX::OverlayInteract(handle)
    {
        _effect = effect;
        _target = effect->fetchIntParam(kParamPickTarget);
        _a = effect->fetchDouble2DParam(kParamPickA);
        _b = effect->fetchDouble2DParam(kParamPickB);
        _trig = effect->fetchIntParam(kParamPickTrigger);
        _show3D = effect->fetchBooleanParam(kParamShow3D);
        _corner = effect->fetchChoiceParam(kParamViewCorner);
        _mode = effect->fetchChoiceParam(kParamMatteMode);
        _primary = effect->fetchDouble3DParam(kParamPrimary);
        _secondary = effect->fetchDouble3DParam(kParamSecondary);
        _mix = effect->fetchDoubleParam(kParamMix);
        _tol = effect->fetchDoubleParam(kParamTolerance);
        _soft = effect->fetchDoubleParam(kParamSoftness3D);
        _lw = effect->fetchDoubleParam(kParamLumaWeight);
        _gain = effect->fetchDoubleParam(kParamGain);
        _sh = effect->fetchDoubleParam(kParamShadows);
        _mid = effect->fetchDoubleParam(kParamMidtones);
        _hi = effect->fetchDoubleParam(kParamHighlights);
        _yaw = effect->fetchDoubleParam(kParamViewYaw);
        _pitch = effect->fetchDoubleParam(kParamViewPitch);
        _zoom = effect->fetchDoubleParam(kParamViewZoom);
        _vsize = effect->fetchDoubleParam(kParamViewSize);
        _tolDisp = effect->fetchChoiceParam(kParamTolDisplay);
        _envDisp = effect->fetchChoiceParam(kParamEnvDisplay);
        _space = effect->fetchChoiceParam(kParamViewSpace);
        for (OFX::Param* q : std::initializer_list<OFX::Param*>{ _target, _show3D, _corner, _mode, _primary, _secondary,
                                                                 _mix, _tol, _soft, _lw, _gain, _sh, _mid, _hi, _yaw,
                                                                 _pitch, _zoom, _vsize, _tolDisp, _envDisp, _space })
            addParamToSlaveTo(q);
    }

    bool draw(const OFX::DrawArgs& args) override
    {
        bool drawn = false;
        const ovl::GLState gl;
        if (viewVisible(args.time)) {
            _frame.capture();
            csv::Settings st = settings(args.time);
            _view.layout(_frame, st.corner, st.size);
            _view.draw(st, _viewDrag || _viewHover, _resizeDrag || _handleHover);
            drawn = true;
        } else {
            _frame.valid = false;
        }
        if (_target->getValue() == 0) { _drag = false; return drawn; }
        const ovl::Pen pen(args.pixelScale);
        const ovl::Colour c = ovl::highlight();
        if (_drag) ovl::rect(pen, c, _x0, _y0, _x1, _y1);
        if (_hasCursor) {
            // picker cursor: a cross with a gap in the middle, so the pixel stays visible
            const double g = 4.0, l = 14.0;
            ovl::line(pen, c, _cx - l * pen.sx, _cy, _cx - g * pen.sx, _cy);
            ovl::line(pen, c, _cx + g * pen.sx, _cy, _cx + l * pen.sx, _cy);
            ovl::line(pen, c, _cx, _cy - l * pen.sy, _cx, _cy - g * pen.sy);
            ovl::line(pen, c, _cx, _cy + g * pen.sy, _cx, _cy + l * pen.sy);
        }
        return true;
    }

    bool penMotion(const OFX::PenArgs& args) override
    {
        double wx = 0, wy = 0;
        const bool inView = viewHit(args, wx, wy);
        if (_resizeDrag) {
            setNow(_vsize, _view.resized(_size0, wx - _vx, wy - _vy));
            requestRedraw();
            return true;
        }
        if (_viewDrag) {
            const double dx = wx - _vx, dy = wy - _vy;
            if (_shift) {
                setNow(_zoom, std::min(20.0, std::max(0.05, _zoom0 * std::exp(dy * 0.01))));
            } else {
                setNow(_yaw, _yaw0 + dx * 0.01);
                setNow(_pitch, std::min(1.55, std::max(-1.55, _pitch0 - dy * 0.01)));
            }
            requestRedraw();
            return true;
        }
        const bool onHandle = inView && _view.onHandle(wx, wy);
        if (inView != _viewHover || onHandle != _handleHover) {
            _viewHover = inView; _handleHover = onHandle;
            requestRedraw();
        }
        if (_target->getValue() == 0) return false;
        _cx = args.penPosition.x; _cy = args.penPosition.y; _hasCursor = !inView;
        if (_drag) { _x1 = _cx; _y1 = _cy; }
        requestRedraw();
        return _drag;
    }

    bool penDown(const OFX::PenArgs& args) override
    {
        double wx = 0, wy = 0;
        if (viewHit(args, wx, wy) && _view.onHandle(wx, wy)) {        // resize
            _resizeDrag = true;
            _vx = wx; _vy = wy;
            _size0 = _view.size;
            requestRedraw();
            return true;
        }
        if (viewHit(args, wx, wy)) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - _lastClick).count() < 0.35) {   // double-click: reset the view
                for (OFX::DoubleParam* d : { _yaw, _pitch, _zoom }) setNow(d, d->getDefault());
                _lastClick = std::chrono::steady_clock::time_point();
                requestRedraw();
                return true;
            }
            _lastClick = now;
            _viewDrag = true;
            _vx = wx; _vy = wy;
            _yaw0 = _yaw->getValue(); _pitch0 = _pitch->getValue(); _zoom0 = _zoom->getValue();
            requestRedraw();
            return true;
        }
        if (_target->getValue() == 0) return false;
        _drag = true;
        _x0 = _x1 = args.penPosition.x;
        _y0 = _y1 = args.penPosition.y;
        requestRedraw();
        return true;
    }

    bool penUp(const OFX::PenArgs& args) override
    {
        if (_viewDrag || _resizeDrag) { _viewDrag = _resizeDrag = false; requestRedraw(); return true; }
        if (!_drag) return false;
        _drag = false;
        _x1 = args.penPosition.x; _y1 = args.penPosition.y;
        // One undo step for the whole pick: the parameters set by the effect when it samples the
        // image (in changedParam, called during setValue) are recorded inside this edit block,
        // so that Ctrl+Z restores the previous colours.
        _effect->beginEditBlock("Pick Colour");
        _a->setValue(_x0, _y0);
        _b->setValue(_x1, _y1);
        _trig->setValue(_trig->getValue() + 1);          // the effect samples the Source
        _effect->endEditBlock();
        requestRedraw();
        return true;
    }

    bool keyDown(const OFX::KeyArgs& args) override
    {
        if (args.keySymbol == kOfxKey_Shift_L || args.keySymbol == kOfxKey_Shift_R) _shift = true;
        if (args.keySymbol == kOfxKey_Escape && _target->getValue() != 0) {
            _drag = false;
            _target->setValue(0);
            requestRedraw();
            return true;
        }
        return false;
    }

    bool keyUp(const OFX::KeyArgs& args) override
    {
        if (args.keySymbol == kOfxKey_Shift_L || args.keySymbol == kOfxKey_Shift_R) _shift = false;
        return false;
    }

    void loseFocus(const OFX::FocusArgs&) override { _hasCursor = false; _shift = false; _viewHover = false; }

private:
    static void setNow(OFX::DoubleParam* p, double v) { p->setValue(v); }

    bool viewVisible(double t) const
    {
        int mode = 0;
        _mode->getValueAtTime(t, mode);
        return _show3D->getValue() && mode != kMatteDifference;
    }

    // Pen position in window pixels (from the last drawn frame) and whether it is in the panel.
    bool viewHit(const OFX::PenArgs& args, double& wx, double& wy) const
    {
        if (!_frame.valid || !viewVisible(args.time)) return false;
        if (!_frame.toWindow(args.penPosition.x, args.penPosition.y, wx, wy)) return false;
        return _view.inside(wx, wy);
    }

    csv::Settings settings(double t) const
    {
        csv::Settings st;
        double r, g, b;
        _primary->getValueAtTime(t, r, g, b);
        st.p.primary[0] = (float)r; st.p.primary[1] = (float)g; st.p.primary[2] = (float)b;
        _secondary->getValueAtTime(t, r, g, b);
        st.p.secondary[0] = (float)r; st.p.secondary[1] = (float)g; st.p.secondary[2] = (float)b;
        st.p.mix = (float)_mix->getValueAtTime(t);
        st.p.tolerance = (float)_tol->getValueAtTime(t);
        st.p.softness3D = (float)_soft->getValueAtTime(t);
        st.p.lumaWeight = (float)_lw->getValueAtTime(t);
        st.p.gain = (float)_gain->getValueAtTime(t);
        st.p.shadows = (float)_sh->getValueAtTime(t);
        st.p.midtones = (float)_mid->getValueAtTime(t);
        st.p.highlights = (float)_hi->getValueAtTime(t);
        _corner->getValue(st.corner);
        st.yaw = _yaw->getValue(); st.pitch = _pitch->getValue(); st.zoom = _zoom->getValue();
        st.size = _vsize->getValue();
        int d = 0;
        _tolDisp->getValue(d); st.tolShaded = d == 1;
        _envDisp->getValue(d); st.envShaded = d == 1;
        _space->getValue(d); st.rgbView = d == 1;
        return st;
    }

    OFX::IntParam* _target;
    OFX::Double2DParam *_a, *_b;
    OFX::IntParam* _trig;
    OFX::BooleanParam* _show3D;
    OFX::ChoiceParam *_corner, *_mode, *_tolDisp, *_envDisp, *_space;
    OFX::Double3DParam *_primary, *_secondary;
    OFX::DoubleParam *_mix, *_tol, *_soft, *_lw, *_gain, *_sh, *_mid, *_hi, *_yaw, *_pitch, *_zoom, *_vsize;
    csv::Frame _frame;
    csv::View _view;
    bool _drag = false, _hasCursor = false, _viewDrag = false, _viewHover = false, _shift = false;
    bool _resizeDrag = false, _handleHover = false;
    double _x0 = 0, _y0 = 0, _x1 = 0, _y1 = 0, _cx = 0, _cy = 0;
    double _vx = 0, _vy = 0, _yaw0 = 0, _pitch0 = 0, _zoom0 = 1, _size0 = 260;
    std::chrono::steady_clock::time_point _lastClick;
};

class PickOverlayDescriptor : public OFX::DefaultEffectOverlayDescriptor<PickOverlayDescriptor, PickOverlay> {};

} // namespace

mDeclarePluginFactory(MasterKeyerFactory, {}, ;);

void MasterKeyerFactory::unload() { cudaReleaseResources(); }

void MasterKeyerFactory::describe(OFX::ImageEffectDescriptor& desc)
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
    desc.setOverlayInteractDescriptor(new PickOverlayDescriptor);
}

namespace {

// Parameters declared after beginTab() go into that tab ("FnOfxParamPropGroupIsTab": Nuke
// shows such a group as a tab of the node panel).
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
D* finish(D* p, OFX::PageParamDescriptor* page, const std::string& label, const char* hint)
{
    p->setLabels(label, label, label);
    p->setHint(hint);
    if (gTab) p->setParent(*gTab);
    page->addChild(*p);
    return p;
}

OFX::DoubleParamDescriptor* dbl(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const std::string& name,
                                const std::string& label, const char* hint, double def, double mn, double mx, double dmn,
                                double dmx)
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

OFX::ChoiceParamDescriptor* choice(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const std::string& name,
                                   const std::string& label, const char* hint, std::initializer_list<const char*> options, int def)
{
    OFX::ChoiceParamDescriptor* p = finish(d.defineChoiceParam(name), page, label, hint);
    for (const char* o : options) p->appendOption(o);
    p->setDefault(def);
    return p;
}

OFX::RGBParamDescriptor* rgb(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, const std::string& name,
                             const std::string& label, const char* hint, double r, double g, double b, double dmn, double dmx)
{
    OFX::RGBParamDescriptor* p = finish(d.defineRGBParam(name), page, label, hint);
    p->setDefault(r, g, b);
    p->setDisplayRange(dmn, dmn, dmn, dmx, dmx, dmx);
    return p;
}

// Colour picked with "Pick Colour": three plain fields (r, g, b), without Nuke's colour swatch
// and picker, which would read the keyed output of the node instead of the Source.
OFX::Double3DParamDescriptor* pickedColour(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page,
                                           const std::string& name, const std::string& label, const char* hint,
                                           double r, double g, double b)
{
    OFX::Double3DParamDescriptor* p = finish(d.defineDouble3DParam(name), page, label, hint);
    p->setDoubleType(OFX::eDoubleTypePlain);
    p->setDimensionLabels("r", "g", "b");
    p->setDefault(r, g, b);
    p->setRange(-1e6, -1e6, -1e6, 1e6, 1e6, 1e6);
    p->setDisplayRange(0, 0, 0, 1, 1, 1);
    return p;
}

OFX::ClipDescriptor* optionalClip(OFX::ImageEffectDescriptor& desc, const char* name)
{
    OFX::ClipDescriptor* c = desc.defineClip(name);
    c->addSupportedComponent(OFX::ePixelComponentRGBA);
    c->addSupportedComponent(OFX::ePixelComponentRGB);
    c->addSupportedComponent(OFX::ePixelComponentAlpha);
    c->setOptional(true);
    c->setSupportsTiles(false);
    return c;
}

const char* kPickHint =
    "Picks this colour in the viewer, on the Key In input if connected, else on the Source (never on the keyed "
    "output): press, then click for one pixel or drag for the average of an area. Press again or Esc to cancel.";

// Button on the line of the parameter declared just before it.
OFX::PushButtonParamDescriptor* sameLineButton(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page,
                                               const std::string& name, OFX::ParamDescriptor* previous,
                                               const char* label, const char* hint)
{
    previous->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
    return finish(d.definePushButtonParam(name), page, label, hint);
}

// "Pick Colour" button on the line of the colour parameter declared just before it.
OFX::PushButtonParamDescriptor* pickButton(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page,
                                           const std::string& name, OFX::ParamDescriptor* colour,
                                           const char* hint = kPickHint)
{
    return sameLineButton(d, page, name, colour, "Pick Colour", hint);
}

OFX::PushButtonParamDescriptor* button(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page,
                                       const std::string& name, const char* label, const char* hint)
{
    return finish(d.definePushButtonParam(name), page, label, hint);
}

void divider(OFX::ParamDescriptor* p) { p->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintDivider, false); }

} // namespace

void MasterKeyerFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);
    optionalClip(desc, kClipKeyIn);
    optionalClip(desc, kClipBG);
    optionalClip(desc, kClipGarbage);
    optionalClip(desc, kClipHoldout);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->addSupportedComponent(OFX::ePixelComponentRGB);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Controls");
    gTab = nullptr;

    // =============================================================== Matte
    beginTab(desc, page, "tabMatte", "Matte");
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
    choice(desc, page, kParamOutput, "Output",
           "Result: keyed foreground composited over the BG input. Without BG, identical to Premult.\n"
           "Premult: despilled and blended foreground, premultiplied by the matte, never composited (to merge it later while keeping the BG connected for From BG).\n"
           "Source CC: the despilled plate only (Auto CC, without the blend), not premultiplied (alpha = matte).\n"
           "Alpha: the matte. Spill: amount of spill removed. Edge: blend region.",
           { "Result", "Premult", "Source CC", "Alpha", "Spill", "Edge" }, kOutResult);
    pickButton(desc, page, kParamPickPrimary, pickedColour(desc, page, kParamPrimary, "Primary",
        "Screen colour: pick it on the screen with Pick Colour. Its largest component gives the screen channel. "
        "The first pick also sets the Secondary and turns Auto CC on; the Spill Colour follows the Primary until "
        "it is changed.",
        0.08, 0.45, 0.12));
    pickButton(desc, page, kParamPickSecond, pickedColour(desc, page, kParamSecondary, "Secondary",
        "Second screen sample, typically on the darkest or least saturated part of the screen. With Mix, the "
        "screen level it represents becomes fully transparent.", 0.06, 0.32, 0.09));
    button(desc, page, kParamUndoPick, "Undo Pick",
           "Undoes the last Pick Colour / Add Colour / Pick Grain (up to 20 picks): Nuke's Ctrl+Z does not undo "
           "the colours a pick sets.");
    dbl(desc, page, kParamMix, "Mix",
        "Balance between the two samples for the transparency level. 0 = only the Primary is fully transparent; "
        "1 = everything at least as 'screen' as the Secondary is fully transparent (cleaner background).",
        0, 0, 1, 0, 1);
    divider(choice(desc, page, kParamScreenSource, "Screen",
           "Colour each pixel is compared to.\nSamples: the Primary colour everywhere.\n"
           "Local: the screen is estimated from the screen areas of the plate and interpolated behind the subject "
           "(uneven screens).", { "Samples", "Local" }, kScreenSamples));
    choice(desc, page, kParamMatteMode, "Matte Mode",
           "Difference: colour-difference keyer - the amount of screen colour in each pixel (screen channel against "
           "the two others). Accurate semi-transparent edges (hair, motion blur).\n"
           "3D: colours are points of a 3D space (luma, Cb, Cr), as in Flame's 3D Keyer. The samples define a "
           "tolerance volume made transparent, surrounded by a softness envelope. Isolates any colour precisely "
           "and keeps colours far from the screen solid.\n"
           "Combined: the larger of the two mattes - 3D keeps the inside of the subject solid, Difference keeps "
           "the edges.", { "Difference", "3D", "Combined" }, kMatteDifference);
    dbl(desc, page, kParamTolerance, "Tolerance",
        "3D: radius of the tolerance volume around the samples (Primary, and Secondary with Mix): colours inside "
        "are fully transparent.", 0.05, 0, 2, 0, 0.3);
    dbl(desc, page, kParamSoftness3D, "3D Softness",
        "3D: width of the softness envelope around the tolerance volume, where the matte goes from transparent to "
        "opaque. Scaled by Gain and Shadows / Midtones / Highlights.", 0.3, 0, 4, 0, 1);
    dbl(desc, page, kParamLumaWeight, "Luma Weight",
        "3D: weight of the luminance in the colour distance. Lower: lighting changes of the screen matter less; "
        "higher: dark or bright colours of the subject are better separated from the screen.", 0.5, 0, 2, 0, 1);
    {
        OFX::BooleanParamDescriptor* b = boolean(desc, page, kParamShow3D, "Show 3D View",
            "3D and Combined modes: shows the colour space in a corner of the viewer while the panel is open - axes "
            "(luma x Luma Weight up, Cb blue, Cr red, dashed: neutral greys), Primary and Secondary, the tolerance "
            "volume (yellow) and the softness envelope (cyan), deformed by Gain and Shadows / Midtones / Highlights. "
            "Drag in it to rotate, Shift + drag to zoom, double-click to reset the view; drag the triangle of its inner "
            "corner to resize it.", true);
        b->setEvaluateOnChange(false);
        b->setAnimates(false);
        b->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
        OFX::ChoiceParamDescriptor* c = choice(desc, page, kParamViewCorner, "Corner", "Corner of the viewer of the 3D view.",
            { "Bottom Left", "Bottom Right", "Top Left", "Top Right" }, 3);
        c->setEvaluateOnChange(false);
        c->setAnimates(false);
        c->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
        OFX::ChoiceParamDescriptor* sp = choice(desc, page, kParamViewSpace, "View Space",
            "Space of the 3D view (display only, the key does not change).\n"
            "Y Cb Cr: the space of the 3D matte - luma x Luma Weight up, Cb and Cr across, neutral greys on the "
            "vertical dashed line, black as a floor.\n"
            "RGB: the RGB cube (R red, G green, B blue axes from black, greys on the dashed diagonal); the volumes "
            "appear tilted and stretched along the grey diagonal.", { "Y Cb Cr", "RGB" }, 0);
        sp->setEvaluateOnChange(false);
        sp->setAnimates(false);
        OFX::ChoiceParamDescriptor* t = choice(desc, page, kParamTolDisplay, "Tolerance Display",
            "3D view: the tolerance volume as a wireframe or as a Gouraud-shaded surface.", { "Wireframe", "Shaded" }, 0);
        t->setEvaluateOnChange(false);
        t->setAnimates(false);
        t->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
        OFX::ChoiceParamDescriptor* e = choice(desc, page, kParamEnvDisplay, "Softness Display",
            "3D view: the softness envelope as a wireframe or as a translucent Gouraud-shaded surface.",
            { "Wireframe", "Shaded" }, 0);
        e->setEvaluateOnChange(false);
        e->setAnimates(false);
    }
    dbl(desc, page, kParamBalance, "Balance",
        "Difference: weight between the two other channels when measuring the screen colour. Green screen: -1 = red "
        "only, +1 = blue only, 0 = equal weights.", 0, -1, 1, -1, 1);
    dbl(desc, page, kParamGain, "Gain", "Strength of the screen measure (overall softness of the matte). Higher: more "
        "transparency.", 1, 0, 10, 0.5, 2);
    dbl(desc, page, kParamShadows, "Shadows",
        "Softness of the matte in the dark parts of the image (by luma). Positive: more transparency; negative: "
        "more opacity. Each unit doubles or halves the Gain there.", 0, -4, 4, -1, 1);
    dbl(desc, page, kParamMidtones, "Midtones", "Softness of the matte in the mid-tones (see Shadows).", 0, -4, 4, -1, 1);
    dbl(desc, page, kParamHighlights, "Highlights", "Softness of the matte in the bright parts (see Shadows).",
        0, -4, 4, -1, 1);
    dbl(desc, page, kParamClipLow, "Clip Black", "Matte values below this become 0 (cleans the background).", 0, 0, 1, 0, 1);
    dbl(desc, page, kParamClipHigh, "Clip White", "Matte values above this become 1 (solidifies the subject).", 1, 0, 1, 0, 1);
    divider(dbl(desc, page, kParamSoftness, "Softness",
        "Gamma of the semi-transparent values. Above 1: thinner edges; below 1: softer, fuller edges.", 1, 0.05, 10, 0.25, 4));
    {
        OFX::IntParamDescriptor* s = finish(desc.defineIntParam(kParamShrink), page, "Shrink / Grow",
            "Erodes (negative) or grows (positive) the matte, in pixels.");
        s->setDefault(0); s->setRange(-100, 100); s->setDisplayRange(-10, 10);
    }
    divider(dbl(desc, page, kParamMatteBlur, "Matte Blur", "Blur of the matte (pixels).", 0, 0, 1000, 0, 20));
    button(desc, page, kParamResetMatte, "Reset Matte",
           "Resets the matte parameters (Screen to Matte Blur). The colours and Mix are kept.");

    // ============================================================= Patches
    beginTab(desc, page, "tabPatches", "Patches");
    divider(button(desc, page, kParamUndoPick2, "Undo Pick",
           "Undoes the last Pick Colour / Add Colour (patches, Primary, Secondary, Spill Colour) or Pick Grain, up to "
           "20 picks: Nuke's Ctrl+Z does not undo the colours a pick sets."));
    for (int k = 0; k < kPatches; ++k) {
        const std::string n = "Patch " + std::to_string(k + 1);
        OFX::BooleanParamDescriptor* en = boolean(desc, page, patchName(k, "Enable").c_str(), n.c_str(),
            "Turns the patch on (done automatically by Pick Colour / Add Colour).", false);
        en->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintNoNewLine, false);
        choice(desc, page, patchName(k, "Type"), "Type",
               "Auto: chosen at the first pick (Black where the matte is mostly transparent, White where it is "
               "mostly opaque).\nBlack: these colours become transparent.\nWhite: these colours become opaque (holes "
               "in the subject).\nEdge Analysis: erodes the semi-transparent edges where these colours are (fringes).",
               { "Auto", "Black", "White", "Edge Analysis" }, kChoiceAuto);
        pickedColour(desc, page, patchName(k, "Low"), "Colour Low",
            "Lowest r, g, b values of the colours of the patch (set by the picks).", 0.5, 0.5, 0.5);
        OFX::Double3DParamDescriptor* hi = pickedColour(desc, page, patchName(k, "High"), "Colour High",
            "Highest r, g, b values of the colours of the patch (set by the picks).", 0.5, 0.5, 0.5);
        OFX::PushButtonParamDescriptor* pk = pickButton(desc, page, patchName(k, "Pick"), hi,
            "Replaces the colours of the patch with a sample of the viewer (Key In if connected, else Source): "
            "press, then click for one pixel or drag over an area. Press again or Esc to cancel.");
        sameLineButton(desc, page, patchName(k, "Add"), pk, "Add Colour",
            "Adds the colours of a new sample to the patch (click or drag in the viewer).");
        dbl(desc, page, patchName(k, "Range"), "Range",
            "Colour distance around the colours of the patch that is fully affected.", 0.05, 0, 2, 0, 0.5);
        dbl(desc, page, patchName(k, "Softness"), "Softness",
            "Extra distance over which the patch fades out.", 0.05, 0, 2, 0, 0.5);
        OFX::PushButtonParamDescriptor* r = button(desc, page, patchName(k, "Reset"), "Reset",
            "Resets all the parameters of this patch to their defaults (patch off).");
        if (k + 1 < kPatches) divider(r);
    }

    // ============================================================= Degrain
    beginTab(desc, page, "tabDegrain", "Degrain");
    button(desc, page, kParamPickGrain, "Pick Grain",
           "Press, then drag over a flat, grainy area in the viewer (Key In if connected, else Source): the grain "
           "size is measured, Size is set and Degrain is turned on. Press again or Esc to cancel.");
    boolean(desc, page, kParamDegrain, "Degrain",
            "Smooths the colour of the plate used to compute the matte (not the image itself), so that grain and "
            "compression noise do not make the matte noisy.", false);
    dbl(desc, page, kParamGrainSize, "Size", "Size of the grain to remove (blur of the colour, pixels).", 2, 0, 100, 0, 10);
    dbl(desc, page, kParamGrainEdges, "Edges",
        "Also smooths the luminance used for the matte: removes more noise but softens the edges.", 0, 0, 1, 0, 1);
    divider(boolean(desc, page, kParamGrainMore, "More", "Doubles the degrain size (very grainy plates).", false));
    button(desc, page, kParamResetDegrain, "Reset Degrain", "Turns Degrain off and resets its parameters.");

    // =============================================================== Spill
    beginTab(desc, page, "tabSpill", "Spill");
    boolean(desc, page, kParamAutoCC, "Auto CC",
            "Removes the screen colour from the subject: the screen channel is limited by the two other channels.", true);
    dbl(desc, page, kParamSpillStr, "Strength", "Spill suppression at the edges of the subject.", 1, 0, 4, 0, 1.5);
    dbl(desc, page, kParamSpillRange, "Range",
        "Distance from the screen (pixels) over which the suppression fades to Interior.", 20, 0, 1e4, 0, 200);
    dbl(desc, page, kParamSpillInt, "Interior",
        "Suppression kept deep inside the subject (relative to Strength). 1 = same everywhere; lower keeps "
        "colours close to the screen colour inside the subject (green eyes, teal clothes).", 0.2, 0, 1, 0, 1);
    dbl(desc, page, kParamHue, "Hue",
        "Moves the suppression towards a neighbouring hue. Green screen: -1 = limited by red (warmer result), "
        "+1 = limited by blue (cooler result).", 0, -1, 1, -1, 1);
    pickButton(desc, page, kParamPickSpill, pickedColour(desc, page, kParamSpillColour, "Spill Colour",
        "Colour of the spill to suppress. It follows the Primary until it is changed; a different colour changes "
        "the suppression (not the key).", 0.08, 0.45, 0.12),
        "Picks the Spill Colour on the Source (the front plate): press, then click for one pixel or drag for the "
        "average of an area. Press again or Esc to cancel.");
    button(desc, page, kParamResetSpill, "Reset Spill",
           "Resets the spill parameters; the Spill Colour goes back to the Primary.");

    // =============================================================== Blend
    beginTab(desc, page, "tabBlend", "Blend");
    dbl(desc, page, kParamEdgeSize, "Edge Size",
        "Width (pixels) of the edge region of the subject treated by the blend. 0 = off.", 0, 0, 1e4, 0, 50);
    dbl(desc, page, kParamEdgeLuma, "Edge Luma", "Darkens (negative) or brightens (positive) the edge region.", 0, -1, 4, -1, 1);
    sameLineButton(desc, page, kParamResetBalance, rgb(desc, page, kParamEdgeBalance, "Edge Balance",
        "Colour shift of the edge region (only the chroma of this colour is used, its luminance is ignored).",
        0, 0, 0, -0.2, 0.2), "Reset", "Resets the Edge Balance only.");
    dbl(desc, page, kParamEdgeFromBG, "From BG",
        "Pushes the colour of the edge region towards the colour of the blurred BG input.", 0, 0, 1, 0, 1);
    divider(dbl(desc, page, kParamBGBlur, "BG Blur", "Blur of the BG (pixels) used by From BG.", 20, 0, 1e4, 0, 200));
    button(desc, page, kParamResetBlend, "Reset Blend", "Resets the blend parameters, except the Edge Balance.");

    // ---- picker state (hidden)
    gTab = nullptr;
    for (const char* n : { kParamPickTarget, kParamPickTrigger }) {
        OFX::IntParamDescriptor* ip = desc.defineIntParam(n);
        ip->setLabels(n, n, n);
        ip->setDefault(0);
        ip->setEvaluateOnChange(false);
        ip->setAnimates(false);
        ip->setIsPersistant(false);
        ip->setIsSecret(true);
        page->addChild(*ip);
    }
    {
        const double defs[4] = { 0.6, 0.35, 1.0, 260.0 };
        const char* names[4] = { kParamViewYaw, kParamViewPitch, kParamViewZoom, kParamViewSize };
        for (int i = 0; i < 4; ++i) {
            OFX::DoubleParamDescriptor* d = desc.defineDoubleParam(names[i]);
            d->setLabels(names[i], names[i], names[i]);
            d->setDefault(defs[i]);
            d->setRange(-1e6, 1e6);
            d->setEvaluateOnChange(false);
            d->setAnimates(false);
            d->setIsSecret(true);
            page->addChild(*d);
        }
    }
    {
        OFX::StringParamDescriptor* h = desc.defineStringParam(kParamPickHistory);
        h->setLabels(kParamPickHistory, kParamPickHistory, kParamPickHistory);
        h->setStringType(OFX::eStringTypeMultiLine);
        h->setDefault("");
        h->setEvaluateOnChange(false);
        h->setAnimates(false);
        h->setIsSecret(true);
        page->addChild(*h);
    }
    {
        OFX::BooleanParamDescriptor* b = desc.defineBooleanParam(kParamKeyPicked);
        b->setLabels(kParamKeyPicked, kParamKeyPicked, kParamKeyPicked);
        b->setDefault(false);
        b->setEvaluateOnChange(false);
        b->setAnimates(false);
        b->setIsSecret(true);
        page->addChild(*b);
    }
    for (const char* n : { kParamPickA, kParamPickB }) {
        OFX::Double2DParamDescriptor* d2 = desc.defineDouble2DParam(n);
        d2->setLabels(n, n, n);
        d2->setDoubleType(OFX::eDoubleTypePlain);
        d2->setDefault(0, 0);
        d2->setEvaluateOnChange(false);
        d2->setAnimates(false);
        d2->setIsPersistant(false);
        d2->setIsSecret(true);
        page->addChild(*d2);
    }
}

OFX::ImageEffect* MasterKeyerFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new MasterKeyerPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static MasterKeyerFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
} // namespace Plugin
} // namespace OFX
