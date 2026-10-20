// GlowPlugin.cpp
// OFX front-end of apGlow_OFX: exponential glow made of many blurs whose
// sizes grow geometrically up to "max size", summed with a progressive colour
// gain ("persistence"). CPU and optional CUDA back-ends.
//
// Inputs : Source, Area (optional, emitting zone), Mask (optional)
#include "ofxsImageEffect.h"
#include "OverlayGL.h"
#include "ofxsLog.h"

#include "GlowCore.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define kPluginName          "apGlow_OFX"
#define kPluginGrouping      "Filter"
#define kPluginIdentifier    "fr.expoglow.expo_GLOW_OFX"   // unchanged since expo_GLOW_OFX: existing Nuke scripts still load
#define kPluginVersionMajor  1
#define kPluginVersionMinor  0

#define kPluginDescription \
    "apGlow_OFX: exponential glow.\n\n" \
    "Sum of several blurs whose size grows geometrically up to Max Size, " \
    "with a progressive colour gain (Persistence) from the smallest to the largest.\n" \
    "Area input: limits the area that emits the glow. Mask input: limits the final effect.\n\n" \
    "Optional CUDA GPU acceleration (NVIDIA cards), automatic fallback to the CPU."

static const char* kClipArea = "Area";
static const char* kClipMask = "Mask";

static const char* kParamUseGPU         = "useGPU";
static const char* kParamGPUStatus      = "gpuStatus";
static const char* kParamIntensity      = "intensity";
static const char* kParamPersistence    = "persistence";
static const char* kParamTint           = "tint";
static const char* kParamEffectOnly     = "effectOnly";
static const char* kParamSrcBlack       = "srcBlack";
static const char* kParamSrcWhite       = "srcWhite";
static const char* kParamWhiteClamp     = "whiteClamp";
static const char* kParamResponse       = "response";
static const char* kParamMaxSize        = "maxSize";
static const char* kParamSizeFalloff    = "sizeFalloff";
static const char* kParamSteps          = "steps";
static const char* kParamSizeScale      = "sizeScale";
static const char* kParamQuality        = "quality";
static const char* kParamFilter         = "filter";
static const char* kParamCropToFormat   = "cropToFormat";
static const char* kParamOverscan       = "overscan";
static const char* kParamEnableOverscan = "enableOverscan";
static const char* kParamInvertArea     = "invertArea";
static const char* kParamInvertMask     = "invertMask";
static const char* kParamMix            = "mix";
static const char* kParamShowOverlay    = "showOverlay";   // label "Show Widgets" (name kept for existing scripts)
static const char* kParamOverlayOffset  = "overlayOffset";

// Layout hint (host extension, not in the base OFX headers): keep next param on the same line.
static const char* kPropLayoutHint = "OfxParamPropLayoutHint";
static const int   kLayoutHintNoNewLine = 2;
static const int   kLayoutHintDivider   = 1;   // horizontal line after the parameter

using namespace expoglow;

// ============================================================ helpers
namespace {

// Runs f(y) for y in [y0, y1) across hardware threads.
template <class F>
void parallelRows(int y0, int y1, F&& f)
{
    const int n = y1 - y0;
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 64 || nt == 1) { for (int y = y0; y < y1; ++y) f(y); return; }
    nt = std::min<unsigned>(nt, (unsigned)n);
    const int chunk = (n + (int)nt - 1) / (int)nt;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t) {
        const int b = y0 + (int)t * chunk, e = std::min(y1, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { for (int y = b; y < e; ++y) f(y); });
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

// Copies `img` into a dense RGBA buffer covering `rect`. Outside the image bounds the
// buffer is black, or (repeatEdges) the edge pixels are repeated as Nuke does outside a bbox.
// RGB images get alpha 1, Alpha images get rgb 0.
void imageToRGBA(const OFX::Image* img, const OfxRectI& rect, std::vector<float>& out, bool repeatEdges = false)
{
    const int w = rect.x2 - rect.x1, h = rect.y2 - rect.y1;
    out.assign((size_t)4 * w * h, 0.f);
    if (!img) return;
    if (img->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
    const OFX::PixelComponentEnum comps = img->getPixelComponents();
    const int nc = componentCount(comps);
    if (nc == 0) return;
    const OfxRectI bd = img->getBounds();
    if (bd.x2 <= bd.x1 || bd.y2 <= bd.y1) return;
    auto readPx = [&](const float* px, float* o) {
        if (nc == 4)      { o[0] = px[0]; o[1] = px[1]; o[2] = px[2]; o[3] = px[3]; }
        else if (nc == 3) { o[0] = px[0]; o[1] = px[1]; o[2] = px[2]; o[3] = 1.f; }
        else              { o[3] = px[0]; }
    };
    if (repeatEdges) {
        parallelRows(rect.y1, rect.y2, [&](int y) {
            const int sy = std::min(std::max(y, bd.y1), bd.y2 - 1);
            const float* row = static_cast<const float*>(img->getPixelAddress(bd.x1, sy));
            if (!row) return;
            float* o = &out[4 * (size_t)(y - rect.y1) * w];
            for (int x = rect.x1; x < rect.x2; ++x, o += 4) {
                const int sx = std::min(std::max(x, bd.x1), bd.x2 - 1);
                readPx(row + (size_t)(sx - bd.x1) * nc, o);
            }
        });
        return;
    }
    const int x1 = std::max(rect.x1, bd.x1), x2 = std::min(rect.x2, bd.x2);
    const int y1 = std::max(rect.y1, bd.y1), y2 = std::min(rect.y2, bd.y2);
    if (x1 >= x2) return;
    parallelRows(y1, y2, [&](int y) {
        const float* px = static_cast<const float*>(img->getPixelAddress(x1, y));
        if (!px) return;
        float* o = &out[4 * ((size_t)(y - rect.y1) * w + (x1 - rect.x1))];
        if (nc == 4) { std::memcpy(o, px, sizeof(float) * 4 * (size_t)(x2 - x1)); return; }
        for (int x = x1; x < x2; ++x, px += nc, o += 4) readPx(px, o);
    });
}

// Single-channel matte from an image: alpha, or luminance for RGB images.
void imageToMatte(const OFX::Image* img, const OfxRectI& rect, bool invert, std::vector<float>& out,
                  bool repeatEdges = false)
{
    std::vector<float> rgba;
    imageToRGBA(img, rect, rgba, repeatEdges);
    const bool rgbOnly = img && img->getPixelComponents() == OFX::ePixelComponentRGB;
    out.resize(rgba.size() / 4);
    for (size_t i = 0; i < out.size(); ++i) {
        const float* p = &rgba[4 * i];
        float m = rgbOnly ? 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2] : p[3];
        m = std::min(1.f, std::max(0.f, m));
        out[i] = invert ? 1.f - m : m;
    }
}

// One pixel of an OFX float image as RGBA (RGB: alpha 1, Alpha: rgb 0).
inline void pixelRGBA(const float* px, int nc, float* o)
{
    if (nc == 4)      { o[0] = px[0]; o[1] = px[1]; o[2] = px[2]; o[3] = px[3]; }
    else if (nc == 3) { o[0] = px[0]; o[1] = px[1]; o[2] = px[2]; o[3] = 1.f; }
    else              { o[0] = o[1] = o[2] = 0.f; o[3] = px[0]; }
}

// Rows [r0, r1) of `rect` (0 = bottom row of rect) of `img`, written densely at `out` with
// `stride` floats per pixel (3: RGB, 1: matte). Same rules as imageToRGBA / imageToMatte:
// black outside the image bounds, or (repeatEdges) the edge pixels repeated.
// matte: alpha, or luminance for RGB images, clamped to [0, 1], optionally inverted.
// Thread-safe: only reads the image.
void readRows(const OFX::Image* img, const OfxRectI& rect, int r0, int r1, bool repeatEdges,
              bool matte, bool invert, float* out)
{
    const int w = rect.x2 - rect.x1;
    const int stride = matte ? 1 : 3;
    const int nc = img ? componentCount(img->getPixelComponents()) : 0;
    const OfxRectI bd = img ? img->getBounds() : OfxRectI{ 0, 0, 0, 0 };
    const bool empty = nc == 0 || bd.x2 <= bd.x1 || bd.y2 <= bd.y1;
    const bool rgbOnly = nc == 3;
    auto put = [&](const float* rgba, float* o) {
        if (!matte) { o[0] = rgba[0]; o[1] = rgba[1]; o[2] = rgba[2]; return; }
        float m = rgbOnly ? 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2] : rgba[3];
        m = std::min(1.f, std::max(0.f, m));
        o[0] = invert ? 1.f - m : m;
    };
    const float zero[4] = { 0.f, 0.f, 0.f, 0.f };
    for (int r = r0; r < r1; ++r) {
        float* o = out + (size_t)(r - r0) * w * stride;
        const int y = rect.y1 + r;
        const float* row = nullptr;
        if (!empty) {
            int sy = y;
            if (repeatEdges) sy = std::min(std::max(y, bd.y1), bd.y2 - 1);
            if (sy >= bd.y1 && sy < bd.y2) row = static_cast<const float*>(img->getPixelAddress(bd.x1, sy));
        }
        // fast path: RGBA image to RGB rows, the row fully inside the image: tight loop
        if (!matte && row && nc == 4 && rect.x1 >= bd.x1 && rect.x2 <= bd.x2) {
            const float* sp = row + (size_t)(rect.x1 - bd.x1) * 4;
            for (int x = 0; x < w; ++x, sp += 4, o += 3) { o[0] = sp[0]; o[1] = sp[1]; o[2] = sp[2]; }
            continue;
        }
        for (int x = rect.x1; x < rect.x2; ++x, o += stride) {
            int sx = x;
            if (repeatEdges) sx = std::min(std::max(x, bd.x1), bd.x2 - 1);
            if (!row || sx < bd.x1 || sx >= bd.x2) { put(zero, o); continue; }
            float px[4];
            pixelRGBA(row + (size_t)(sx - bd.x1) * nc, nc, px);
            put(px, o);
        }
    }
}

inline OfxRectD expandRect(const OfxRectD& r, double e) { return { r.x1 - e, r.y1 - e, r.x2 + e, r.y2 + e }; }
inline OfxRectI expandRect(const OfxRectI& r, int e)    { return { r.x1 - e, r.y1 - e, r.x2 + e, r.y2 + e }; }
inline bool emptyRect(const OfxRectD& r) { return r.x2 <= r.x1 || r.y2 <= r.y1; }
inline OfxRectD intersectRect(const OfxRectD& a, const OfxRectD& b)
{
    OfxRectD r = { std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2) };
    if (emptyRect(r)) r = { 0, 0, 0, 0 };
    return r;
}
inline OfxRectI intersectRect(const OfxRectI& a, const OfxRectI& b)
{
    OfxRectI r = { std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2) };
    if (r.x2 <= r.x1 || r.y2 <= r.y1) r = { 0, 0, 0, 0 };
    return r;
}
inline OfxRectI unionRect(const OfxRectI& a, const OfxRectI& b)
{
    if (a.x2 <= a.x1 || a.y2 <= a.y1) return b;
    if (b.x2 <= b.x1 || b.y2 <= b.y1) return a;
    return { std::min(a.x1, b.x1), std::min(a.y1, b.y1), std::max(a.x2, b.x2), std::max(a.y2, b.y2) };
}

} // namespace

// ================================================================== plugin
class ExpoGlowPlugin : public OFX::ImageEffect
{
public:
    explicit ExpoGlowPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;

private:
    GlowSettings settingsAt(double time, double renderScale);
    bool glowArea(double time, OfxRectD& area);
    void updateParamsState();

    OFX::Clip* _dstClip;
    OFX::Clip* _srcClip;
    OFX::Clip* _areaClip;
    OFX::Clip* _maskClip;

    OFX::BooleanParam* _useGPU;
    OFX::DoubleParam*  _intensity;
    OFX::RGBParam*     _persistence;
    OFX::RGBParam*     _tint;
    OFX::BooleanParam* _effectOnly;
    OFX::DoubleParam*  _srcBlack;
    OFX::DoubleParam*  _srcWhite;
    OFX::BooleanParam* _whiteClamp;
    OFX::DoubleParam*  _response;
    OFX::DoubleParam*  _maxSize;
    OFX::DoubleParam*  _sizeFalloff;
    OFX::IntParam*     _steps;
    OFX::DoubleParam*  _sizeScale;
    OFX::IntParam*     _quality;
    OFX::ChoiceParam*  _filter;
    OFX::BooleanParam* _cropToFormat;
    OFX::DoubleParam*  _overscan;
    OFX::BooleanParam* _enableOverscan;
    OFX::BooleanParam* _invertArea;
    OFX::BooleanParam* _invertMask;
    OFX::DoubleParam*  _mix;
};

ExpoGlowPlugin::ExpoGlowPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _dstClip  = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip  = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _areaClip = fetchClip(kClipArea);
    _maskClip = fetchClip(kClipMask);

    _useGPU         = fetchBooleanParam(kParamUseGPU);
    _intensity      = fetchDoubleParam(kParamIntensity);
    _persistence    = fetchRGBParam(kParamPersistence);
    _tint           = fetchRGBParam(kParamTint);
    _effectOnly     = fetchBooleanParam(kParamEffectOnly);
    _srcBlack       = fetchDoubleParam(kParamSrcBlack);
    _srcWhite       = fetchDoubleParam(kParamSrcWhite);
    _whiteClamp     = fetchBooleanParam(kParamWhiteClamp);
    _response       = fetchDoubleParam(kParamResponse);
    _maxSize        = fetchDoubleParam(kParamMaxSize);
    _sizeFalloff    = fetchDoubleParam(kParamSizeFalloff);
    _steps          = fetchIntParam(kParamSteps);
    _sizeScale      = fetchDoubleParam(kParamSizeScale);
    _quality        = fetchIntParam(kParamQuality);
    _filter         = fetchChoiceParam(kParamFilter);
    _cropToFormat   = fetchBooleanParam(kParamCropToFormat);
    _overscan       = fetchDoubleParam(kParamOverscan);
    _enableOverscan = fetchBooleanParam(kParamEnableOverscan);
    _invertArea     = fetchBooleanParam(kParamInvertArea);
    _invertMask     = fetchBooleanParam(kParamInvertMask);
    _mix            = fetchDoubleParam(kParamMix);
    updateParamsState();
}

void ExpoGlowPlugin::updateParamsState()
{
    bool crop = false, ov = true;
    _cropToFormat->getValue(crop);
    _enableOverscan->getValue(ov);
    _enableOverscan->setEnabled(!crop);
    _overscan->setEnabled(!crop && ov);
    _useGPU->setEnabled(cudaIsCompiledIn());
}

void ExpoGlowPlugin::changedParam(const OFX::InstanceChangedArgs&, const std::string& name)
{
    if (name == kParamCropToFormat || name == kParamEnableOverscan) updateParamsState();
}

GlowSettings ExpoGlowPlugin::settingsAt(double t, double renderScale)
{
    GlowSettings g;
    g.intensity = (float)_intensity->getValueAtTime(t);
    double r, gg, b;
    _persistence->getValueAtTime(t, r, gg, b);
    g.persistence[0] = (float)r; g.persistence[1] = (float)gg; g.persistence[2] = (float)b;
    _tint->getValueAtTime(t, r, gg, b);
    g.tint[0] = (float)r; g.tint[1] = (float)gg; g.tint[2] = (float)b;
    g.maxSize = (float)_maxSize->getValueAtTime(t);
    g.sizeFalloff = (float)_sizeFalloff->getValueAtTime(t);
    g.steps = std::max(1, _steps->getValueAtTime(t));
    g.sizeScale = (float)_sizeScale->getValueAtTime(t);
    g.quality = (float)_quality->getValueAtTime(t);
    int f = kFilterGaussian;
    _filter->getValueAtTime(t, f);
    g.filter = f;
    g.renderScale = (float)renderScale;
    return g;
}

// Glow area = source bbox grown by the glow, then cropped to the format or to format + overscan.
// Returns true when a crop applies (the result then gets a 1 px black border, like a Nuke Crop
// with "black outside", so that downstream nodes see black beyond the bbox).
bool ExpoGlowPlugin::glowArea(double t, OfxRectD& area)
{
    const OfxRectD src = _srcClip->getRegionOfDefinition(t);
    const int ext = planExtent(planGlow(settingsAt(t, 1.0)));
    area = expandRect(src, ext);
    const OfxPointD off = getProjectOffset(), size = getProjectExtent();
    const OfxRectD format = { off.x, off.y, off.x + size.x, off.y + size.y };
    if (_cropToFormat->getValueAtTime(t)) {
        area = intersectRect(area, format);
        return true;
    }
    if (_enableOverscan->getValueAtTime(t)) {
        area = intersectRect(area, expandRect(format, _overscan->getValueAtTime(t)));
        return true;
    }
    return false;
}

bool ExpoGlowPlugin::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if (!_srcClip->isConnected()) return false;
    OfxRectD area;
    const bool cropped = glowArea(args.time, area);
    rod = (cropped && !emptyRect(area)) ? expandRect(area, 1.0) : area;
    return true;
}

void ExpoGlowPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois)
{
    const int ext = planExtent(planGlow(settingsAt(args.time, 1.0)));
    const OfxRectD roi = expandRect(args.regionOfInterest, ext);
    rois.setRegionOfInterest(*_srcClip, roi);
    if (_areaClip->isConnected()) rois.setRegionOfInterest(*_areaClip, roi);
    if (_maskClip->isConnected()) rois.setRegionOfInterest(*_maskClip, args.regionOfInterest);
}

void ExpoGlowPlugin::render(const OFX::RenderArguments& args)
{
    const double t = args.time;
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(t));
    if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (dst->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);

    const OfxRectI win = dst->getBounds();
    const int ww = win.x2 - win.x1, wh = win.y2 - win.y1;
    if (ww <= 0 || wh <= 0) return;

    std::unique_ptr<const OFX::Image> src(_srcClip->isConnected() ? _srcClip->fetchImage(t) : nullptr);
    std::unique_ptr<const OFX::Image> area(_areaClip->isConnected() ? _areaClip->fetchImage(t) : nullptr);
    std::unique_ptr<const OFX::Image> mask(_maskClip->isConnected() ? _maskClip->fetchImage(t) : nullptr);
    for (const OFX::Image* im : { src.get(), area.get(), mask.get() })
        if (im && im->getPixelDepth() != OFX::eBitDepthFloat) OFX::throwSuiteStatusException(kOfxStatErrImageFormat);

    // ---- glow plan at the render scale
    const double scale = 0.5 * (args.renderScale.x + args.renderScale.y);
    const std::vector<StepPlan> plan = planGlow(settingsAt(t, scale));
    const int ext = planExtent(plan);

    // ---- working buffer: output window + source pixels whose glow reaches it
    OfxRectI buf = win;
    if (src) buf = unionRect(win, intersectRect(expandRect(win, ext), src->getBounds()));
    const int bw = buf.x2 - buf.x1, bh = buf.y2 - buf.y1;

    PrepParams pp;
    pp.width = bw; pp.height = bh;
    pp.black = (float)_srcBlack->getValueAtTime(t);
    pp.white = (float)_srcWhite->getValueAtTime(t);
    pp.whiteClamp = _whiteClamp->getValueAtTime(t);
    pp.gamma = (float)_response->getValueAtTime(t);

    const bool invertArea = _invertArea->getValueAtTime(t);
    const bool invertMask = _invertMask->getValueAtTime(t);
    const bool effectOnly = _effectOnly->getValueAtTime(t);
    const float mix = (float)_mix->getValueAtTime(t);

    // pixels outside the glow area (the 1 px crop border) are black
    OfxRectI inner = win;
    if (src) {
        OfxRectD a;
        if (glowArea(t, a)) {
            inner = { (int)std::floor(a.x1 * args.renderScale.x + 0.5), (int)std::floor(a.y1 * args.renderScale.y + 0.5),
                      (int)std::floor(a.x2 * args.renderScale.x + 0.5), (int)std::floor(a.y2 * args.renderScale.y + 0.5) };
        }
    }

    // ---- composite of output rows [r0, r1) (0 = bottom row of win). glow: RGB of those rows,
    // `gpx` floats per pixel, `grow` floats per row. Source black outside its bbox.
    const int nc = componentCount(dst->getPixelComponents());
    const int snc = src ? componentCount(src->getPixelComponents()) : 0;
    const OfxRectI sb = src ? src->getBounds() : OfxRectI{ 0, 0, 0, 0 };
    auto compositeRows = [&](int r0, int r1, const float* glow, int gpx, size_t grow) {
        std::vector<float> maskRow(mask ? (size_t)ww : 0);
        for (int r = r0; r < r1; ++r) {
            const int y = win.y1 + r;
            float* px = static_cast<float*>(dst->getPixelAddress(win.x1, y));
            if (!px) continue;
            if (mask) readRows(mask.get(), win, r, r + 1, false, true, invertMask, maskRow.data());
            const float* srow = (snc && y >= sb.y1 && y < sb.y2) ? static_cast<const float*>(src->getPixelAddress(sb.x1, y)) : nullptr;
            const float* g = glow + (size_t)(r - r0) * grow;
            for (int x = win.x1; x < win.x2; ++x, px += nc, g += gpx) {
                if (x < inner.x1 || x >= inner.x2 || y < inner.y1 || y >= inner.y2) {
                    for (int c = 0; c < nc; ++c) px[c] = 0.f;
                    continue;
                }
                float s[4] = { 0.f, 0.f, 0.f, 0.f };
                if (srow && x >= sb.x1 && x < sb.x2) pixelRGBA(srow + (size_t)(x - sb.x1) * snc, snc, s);
                float m = mix;
                if (mask) m *= maskRow[x - win.x1];
                float o[4];
                for (int c = 0; c < 3; ++c) {
                    const float fx = (effectOnly ? 0.f : s[c]) + g[c];
                    o[c] = s[c] + (fx - s[c]) * m;
                }
                o[3] = s[3];
                if (nc == 4)      { px[0] = o[0]; px[1] = o[1]; px[2] = o[2]; px[3] = o[3]; }
                else if (nc == 3) { px[0] = o[0]; px[1] = o[1]; px[2] = o[2]; }
                else if (nc == 1) { px[0] = o[3]; }
            }
        }
    };

    // ---- GPU: rows go straight from the source into the transfer memory (RGB only), and
    // only the output window comes back, composited straight into the output image.
    // Like Nuke, the blurs see the source (and area) with their edge pixels repeated
    // beyond their bounding box, so the glow does not fade at the image borders.
    if (_useGPU->getValueAtTime(t) && cudaIsCompiledIn()) {
        const RowReader rSrc = [&](int r0, int r1, float* o) { readRows(src.get(), buf, r0, r1, true, false, false, o); };
        const RowReader rArea = [&](int r0, int r1, float* o) { readRows(area.get(), buf, r0, r1, true, true, invertArea, o); };
        const RowWriter wGlow = [&](int r0, int r1, const float* g) { compositeRows(r0, r1, g, 3, (size_t)3 * ww); };
        std::string err;
        if (computeGlowCUDARows(pp, plan, rSrc, area ? &rArea : nullptr, win.x1 - buf.x1, win.y1 - buf.y1, ww, wh, wGlow, err))
            return;
        OFX::Log::warning(true, "apGlow_OFX: GPU render failed (%s), falling back to CPU", err.c_str());
    }

    // ---- CPU
    std::vector<float> srcBuf, areaBuf;
    std::unique_ptr<float[]> glowMem(new float[(size_t)4 * bw * bh]);   // fully written by the back-end
    float* const glowBuf = glowMem.get();
    imageToRGBA(src.get(), buf, srcBuf, true);
    if (area) imageToMatte(area.get(), buf, invertArea, areaBuf, true);
    computeGlowCPU(pp, plan, srcBuf.data(), area ? areaBuf.data() : nullptr, glowBuf);
    const float* g0 = glowBuf + 4 * ((size_t)(win.y1 - buf.y1) * bw + (win.x1 - buf.x1));
    parallelRows(0, wh, [&](int r) { compositeRows(r, r + 1, g0 + 4 * (size_t)r * bw, 4, (size_t)4 * bw); });
}

// ================================================================= factory
// ================================================================== viewer overlay
// Drawn while the node panel is open (Show Widgets on): one circle per glow step (diameter =
// blur size of the step, Size Scale included), the largest one solid. Drag its handle to set
// Max Size. The widget stays at the centre of the format.
namespace {

template <class P, class... V>
void setNow(P* p, double t, V... v)
{
    if (p->getNumKeys() > 0) p->setValueAtTime(t, v...);
    else p->setValue(v...);
}

class GlowOverlay : public OFX::OverlayInteract
{
public:
    GlowOverlay(OfxInteractHandle handle, OFX::ImageEffect* effect) : OFX::OverlayInteract(handle)
    {
        _effect = effect;
        _show = effect->fetchBooleanParam(kParamShowOverlay);
        _offset = effect->fetchDouble2DParam(kParamOverlayOffset);
        _maxSize = effect->fetchDoubleParam(kParamMaxSize);
        _falloff = effect->fetchDoubleParam(kParamSizeFalloff);
        _steps = effect->fetchIntParam(kParamSteps);
        _scale = effect->fetchDoubleParam(kParamSizeScale);
        _src = effect->fetchClip(kOfxImageEffectSimpleSourceClipName);
        for (OFX::Param* q : std::initializer_list<OFX::Param*>{ _show, _offset, _maxSize, _falloff, _steps, _scale })
            addParamToSlaveTo(q);
    }

    bool draw(const OFX::DrawArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        const double t = args.time;
        double cx, cy, par;
        centre(t, cx, cy, par);
        const ovl::Pen pen(args.pixelScale);
        const ovl::GLState gl;
        const ovl::Colour base = ovl::baseColour(*this);
        GlowSettings g;
        g.maxSize = (float)_maxSize->getValueAtTime(t);
        g.sizeFalloff = (float)_falloff->getValueAtTime(t);
        g.steps = std::max(1, _steps->getValueAtTime(t));
        g.sizeScale = (float)_scale->getValueAtTime(t);
        g.renderScale = 1.f;
        for (int i = 1; i < g.steps; ++i) {
            const double r = 0.5 * stepSize(g, i);
            ovl::circle(pen, base, cx, cy, r * par, r, 1.f, true);
        }
        const double R = 0.5 * stepSize(g, g.steps);
        const ovl::Colour c = _drag == kRadius || _hover == kRadius ? ovl::highlight() : base;
        ovl::circle(pen, c, cx, cy, R * par, R, 2.f);
        ovl::handle(pen, c, cx + R * par, cy);
        ovl::cross(pen, base, cx, cy);
        return true;
    }

    bool penMotion(const OFX::PenArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        const double x = args.penPosition.x, y = args.penPosition.y;
        if (_drag == kNone) {
            const Part h = hit(args);
            if (h != _hover) { _hover = h; requestRedraw(); }
            return false;
        }
        {
            double cx, cy, par;
            centre(args.time, cx, cy, par);
            const double dx = (x - cx) / par, dy = y - cy, sc = _scale->getValueAtTime(args.time);
            if (sc > 1e-6) setNow(_maxSize, args.time, 2.0 * std::sqrt(dx * dx + dy * dy) / sc);
        }
        requestRedraw();
        return true;
    }

    bool penDown(const OFX::PenArgs& args) override
    {
        if (!_show->getValueAtTime(args.time)) return false;
        _drag = hit(args);
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
    enum Part { kNone, kCentre, kRadius };

    void centre(double t, double& cx, double& cy, double& par) const
    {
        const OfxPointD off = _effect->getProjectOffset(), ext = _effect->getProjectExtent();
        (void)t;                                     // fixed at the centre of the format
        cx = off.x + 0.5 * ext.x;
        cy = off.y + 0.5 * ext.y;
        par = _src->isConnected() ? _src->getPixelAspectRatio() : 1.0;
        if (!(par > 0.0)) par = 1.0;
    }

    Part hit(const OFX::PenArgs& args) const
    {
        const double t = args.time, x = args.penPosition.x, y = args.penPosition.y;
        double cx, cy, par;
        centre(t, cx, cy, par);
        const ovl::Pen pen(args.pixelScale);
        const double R = 0.5 * _maxSize->getValueAtTime(t) * _scale->getValueAtTime(t);
        if (pen.screenDist(x, y, cx + R * par, cy) < pen.hitRadius()) return kRadius;
        return kNone;
    }

    OFX::BooleanParam* _show;
    OFX::Double2DParam* _offset;
    OFX::DoubleParam* _maxSize;
    OFX::DoubleParam* _falloff;
    OFX::IntParam* _steps;
    OFX::DoubleParam* _scale;
    OFX::Clip* _src;
    Part _drag = kNone, _hover = kNone;
    double _dragX = 0, _dragY = 0, _dragOffX = 0, _dragOffY = 0;
};

class GlowOverlayDescriptor : public OFX::DefaultEffectOverlayDescriptor<GlowOverlayDescriptor, GlowOverlay> {};

} // namespace

mDeclarePluginFactory(ExpoGlowFactory, {}, ;);

void ExpoGlowFactory::unload() { cudaReleaseResources(); }

void ExpoGlowFactory::describe(OFX::ImageEffectDescriptor& desc)
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
    desc.setOverlayInteractDescriptor(new GlowOverlayDescriptor);
}

namespace {

template <class D>
D* finish(D* p, OFX::PageParamDescriptor* page, OFX::GroupParamDescriptor* group, const char* label, const char* hint)
{
    p->setLabels(label, label, label);
    p->setHint(hint);
    if (group) p->setParent(*group);
    page->addChild(*p);
    return p;
}

OFX::DoubleParamDescriptor* dbl(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, OFX::GroupParamDescriptor* g,
                                const char* name, const char* label, const char* hint,
                                double def, double mn, double mx, double dmn, double dmx)
{
    OFX::DoubleParamDescriptor* p = finish(d.defineDoubleParam(name), page, g, label, hint);
    p->setDefault(def); p->setRange(mn, mx); p->setDisplayRange(dmn, dmx);
    return p;
}

OFX::BooleanParamDescriptor* boolean(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, OFX::GroupParamDescriptor* g,
                                     const char* name, const char* label, const char* hint, bool def)
{
    OFX::BooleanParamDescriptor* p = finish(d.defineBooleanParam(name), page, g, label, hint);
    p->setDefault(def);
    return p;
}

OFX::RGBParamDescriptor* rgb(OFX::ImageEffectDescriptor& d, OFX::PageParamDescriptor* page, OFX::GroupParamDescriptor* g,
                             const char* name, const char* label, const char* hint)
{
    OFX::RGBParamDescriptor* p = finish(d.defineRGBParam(name), page, g, label, hint);
    p->setDefault(1, 1, 1);
    p->setDisplayRange(0, 0, 0, 4, 4, 4);
    return p;
}

} // namespace

void ExpoGlowFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum)
{
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);

    OFX::ClipDescriptor* area = desc.defineClip(kClipArea);
    area->addSupportedComponent(OFX::ePixelComponentRGBA);
    area->addSupportedComponent(OFX::ePixelComponentRGB);
    area->addSupportedComponent(OFX::ePixelComponentAlpha);
    area->setOptional(true);
    area->setSupportsTiles(false);

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


    // ---- GPU : first line, GPU name on the same line
    {
        OFX::BooleanParamDescriptor* b = boolean(desc, page, nullptr, kParamUseGPU, "Use GPU (CUDA)",
            "Renders the glow on the NVIDIA GPU. Falls back to the CPU automatically if CUDA is not available.", true);
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
    }

    // Sections are separated by horizontal lines (layout hint "divider" after the last
    // parameter of a section) instead of collapsible groups.
    auto divider = [](OFX::ParamDescriptor* p) { p->getPropertySet().propSetInt(kPropLayoutHint, kLayoutHintDivider, false); };

    // ---- glow
    dbl(desc, page, nullptr, kParamIntensity, "Intensity", "Glow intensity multiplier.", 1, 0, 1e6, 0, 10);
    rgb(desc, page, nullptr, kParamPersistence, "Persistence",
        "Colour gain of the largest blur. Affects the other blurs progressively, the smallest one very little.\n"
        "Use it as an inverse falloff or as a progressive colour tint.");
    rgb(desc, page, nullptr, kParamTint, "Tint", "Overall colour multiplier of the glow.");
    divider(boolean(desc, page, nullptr, kParamEffectOnly, "Effect Only",
                    "Outputs the glow only (source alpha kept).", true));

    // ---- source response
    dbl(desc, page, nullptr, kParamSrcBlack, "Src Black",
        "Black point of the source luminance: nothing glows below it.", 0, -1e6, 1e6, 0, 4);
    dbl(desc, page, nullptr, kParamSrcWhite, "Src White", "White point of the source luminance.", 1, -1e6, 1e6, 0, 4);
    dbl(desc, page, nullptr, kParamResponse, "Response",
        "Gamma on the source luminance (shape of the response curve).", 1, 0, 1e3, 0, 4);
    divider(boolean(desc, page, nullptr, kParamWhiteClamp, "Clamp", "Clamps the luminance to 1 after the white point.", false));

    // ---- size
    dbl(desc, page, nullptr, kParamSizeScale, "Size Scale",
        "Multiplies the size of every blur (fine match with the Nuke Blur). 1 = reference.",
        1, 0, 100, 0.25, 4);
    dbl(desc, page, nullptr, kParamMaxSize, "Max Size", "Size of the largest blur (pixels).", 2000, 0, 1e5, 0, 10000);
    dbl(desc, page, nullptr, kParamSizeFalloff, "Size Falloff",
        "Distribution of the blur sizes.\nLower: more diffuse glow (larger blurs from the start).\n"
        "Higher: tighter, more defined glow.", 1, 1e-4, 1e4, 0, 10);
    {
        OFX::IntParamDescriptor* s = finish(desc.defineIntParam(kParamSteps), page, nullptr, "Steps",
                                            "Number of blurs added together.");
        s->setDefault(10); s->setRange(1, 256); s->setDisplayRange(1, 32);
    }
    {
        OFX::ChoiceParamDescriptor* f = finish(desc.defineChoiceParam(kParamFilter), page, nullptr, "Filter",
                                               "Blur shape, as in the Nuke Blur node.");
        f->appendOption("box");
        f->appendOption("triangle");
        f->appendOption("quadratic");
        f->appendOption("gaussian");
        f->setDefault(kFilterGaussian);
    }
    {
        OFX::IntParamDescriptor* q = finish(desc.defineIntParam(kParamQuality), page, nullptr, "Quality",
            "As in the Nuke Blur: above this size, the blur is computed on a reduced image then "
            "scaled back up, which softens the filter edges (visible with box). 15 = Nuke default. "
            "0 = exact filter.");
        q->setDefault(15);
        q->setRange(0, 1000);
        q->setDisplayRange(0, 100);
        divider(q);
    }

    // ---- bounding box
    boolean(desc, page, nullptr, kParamCropToFormat, "Crop to Format", "Crops the result to the project format.", false);
    dbl(desc, page, nullptr, kParamOverscan, "Overscan", "Margin around the format (pixels) in which the glow is kept.",
        100, 0, 1e5, 0, 1000);
    divider(boolean(desc, page, nullptr, kParamEnableOverscan, "Enable Overscan",
        "Limits the bbox to the format + overscan. Off: the bbox covers the whole glow extent.", true));

    // ---- area / mask
    boolean(desc, page, nullptr, kParamInvertArea, "Invert Area", "Inverts the Area input (area that emits the glow).", false);
    boolean(desc, page, nullptr, kParamInvertMask, "Invert Mask", "Inverts the Mask input.", false);
    divider(dbl(desc, page, nullptr, kParamMix, "Mix", "Effect mix.", 1, 0, 1, 0, 1));

    // ---- viewer
    {
        OFX::BooleanParamDescriptor* b = boolean(desc, page, nullptr, kParamShowOverlay, "Show Widgets",
            "Shows the glow sizes in the viewer while the panel is open: one circle per step, the largest one "
            "solid (drag its handle to set Max Size). The widget stays at the centre of the format.", true);
        b->setEvaluateOnChange(false);
        b->setAnimates(false);
        OFX::Double2DParamDescriptor* o = desc.defineDouble2DParam(kParamOverlayOffset);
        o->setLabels("Overlay Offset", "Overlay Offset", "Overlay Offset");
        o->setHint("Position of the viewer widget, relative to the centre of the format.");
        o->setDefault(0, 0);
        o->setDoubleType(OFX::eDoubleTypePlain);
        o->setEvaluateOnChange(false);
        o->setAnimates(false);
        o->setIsSecret(true);
        page->addChild(*o);
        divider(b);
    }

    // ---- credits, at the bottom of the panel under a separator
    {
        OFX::StringParamDescriptor* s = desc.defineStringParam("creditsLine1");
        s->setLabels("", "", "");
        s->setStringType(OFX::eStringTypeLabel);
        s->setDefault("Based on apGlow v1.2 - adrianpueyo.com, 2015-2022");
        s->setIsPersistant(false);
        s->setEvaluateOnChange(false);
        s->setAnimates(false);
        page->addChild(*s);
    }
}

OFX::ImageEffect* ExpoGlowFactory::createInstance(OfxImageEffectHandle handle, OFX::ContextEnum)
{
    return new ExpoGlowPlugin(handle);
}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids)
{
    static ExpoGlowFactory p(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
    ids.push_back(&p);
}
} // namespace Plugin
} // namespace OFX
