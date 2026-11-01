// ChromaAberrations_OFX - physically based optical aberrations, OpenFX plugin for Nuke
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).
//
// The plugin is a thin layer over the engine (src/core, src/cuda):
//  - the parameters describe a parametric lens (Seidel / Zernike), its diaphragm and the sensor;
//  - the lens plan (PSF bank, distortion and illumination maps, spectral table) depends only on
//    the optical parameters and the image size: it is cached per instance and recomputed only
//    when one of them changes (in a worker thread, with a progress bar);
//  - each frame is rendered on the GPU (CUDA + cuFFT) when available, else on the CPU, with
//    exactly the same pipeline (see render.h).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fstream>
#include <map>
#include <mutex>
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dirent.h>
#  include <dlfcn.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "camera.h"
#include "lensplan.h"
#include "lensmodel.h"
#include "reallens.h"
#include "render.h"
#include "render_gpu.h"
#include "views.h"

using namespace OFX;

#define kPluginName "ChromaAberrations_OFX"
#define kPluginGrouping "Filter"
#define kPluginIdentifier "community.openfx.ChromaAberrations_OFX"
#define kPluginVersionMajor 0
#define kPluginVersionMinor 4
#define kPluginDescription \
    "Physically based lens aberrations.\n" \
    "The lens is described by its wavefront (Seidel aberrations and Zernike terms), its diaphragm and its sensor. " \
    "For every wavelength (400-700 nm) the plugin computes the point spread function (PSF) by diffraction " \
    "(FFT of the pupil) on a grid over the image, then renders the plate: spectral upsampling of each pixel, " \
    "distortion and lateral colour per wavelength, blur varying over the field (exact interpolation of the PSF grid), " \
    "relative illumination, and back to RGB with the CIE 1931 colour matching functions.\n" \
    "Two lens models: Parametric (Seidel aberrations and Zernike terms, in waves at 550 nm, at the edge of the pupil and in the image corner) " \
    "or Real Lens: an optical prescription (built-in classics or a .lens file) ray traced wavelength by wavelength through its glasses.\n" \
    "The lens is computed once per setting (progress bar), then each frame costs a few FFT convolutions " \
    "(CUDA GPU when available).\n" \
    "Images must be linear (ACEScg by default, see Spectral > Colour Space)."

// ---- clips
#define kClipMask "Mask"

// ---- Lens page
#define kParamPreset "preset"
#define kParamLensModel "lensModel"
#define kParamLens "lens"
#define kParamLensFile "lensFile"
#define kParamFocusDistance "focusDistance"
#define kParamFNumber "fNumber"
#define kParamFocal "focalLength"
#define kParamFilmback "filmbackWidth"
#define kParamCenter "opticalCenter"
#define kParamCenterOnImage "centerOnImage"
#define kParamBlades "blades"
#define kParamRoundness "bladeRoundness"
#define kParamRotation "bladeRotation"
#define kParamPupilAspect "pupilAspect"
#define kParamVignetting "mechanicalVignetting"
#define kParamVignettingStart "mechanicalVignettingStart"
#define kParamNatural "naturalVignetting"
// ---- Aberrations page
#define kParamSpherical "spherical"
#define kParamComa "coma"
#define kParamAstig "astigmatism"
#define kParamCurvature "fieldCurvature"
#define kParamDistortion "distortion"
#define kParamDefocus "defocus"
#define kParamAxialCA "axialCA"
#define kParamLateralCA "lateralCA"
#define kParamSpheroChrom "spherochromatism"
#define kParamZernike "zernike"
#define kParamAmount "aberrationAmount"
#define kParamDistortionAmount "distortionAmount"
#define kParamLateralAmount "lateralColourAmount"
// ---- Spectral page
#define kParamColorSpace "colorSpace"
#define kParamCamera "camera"
#define kParamCameraFile "cameraFile"
#define kParamWlPreview "wavelengthsPreview"
#define kParamWlFinal "wavelengthsFinal"
// ---- Render page
#define kParamQuality "quality"
#define kParamGridPreview "gridPreview"
#define kParamGridFinal "gridFinal"
#define kParamMaxPsf "maxPsfSize"
#define kParamAutoPsf "autoPsfSize"
#define kParamGridMode "finalGridMode"
#define kParamGridTolerance "gridTolerance"
#define kParamUseGPU "useGPU"
#define kParamGPUStatus "gpuStatus"
// ---- Output page
#define kParamView "view"
#define kParamPsfZoom "psfZoom"
#define kParamPsfBackground "psfMapBackground"
#define kParamBlur "enableBlur"
#define kParamDistort "enableDistortion"
#define kParamIllum "enableVignetting"
#define kParamMix "mix"
#define kParamMaskInvert "maskInvert"

enum ViewEnum { eViewImage = 0, eViewPsfMap, eViewTestChart };
enum LensModelEnum { eLensParametric = 0, eLensReal };
enum QualityEnum { eQualityPreview = 0, eQualityFinal, eQualityAuto };

static bool gIsNuke = false;

// ----------------------------------------------------------------------------- helpers

static void parallelFor(int n, const std::function<void(int, int)>& f)
{
    const int nt = std::max( 1, std::min( (int)std::thread::hardware_concurrency(), n / 64 ) );
    if (nt <= 1) {
        f(0, n);
        return;
    }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back(f, (int)( (long long)n * t / nt ), (int)( (long long)n * (t + 1) / nt ));
    }
    for (std::thread& t : th) {
        t.join();
    }
}

static std::string gpuDescription()
{
    std::string name;
    if ( ca::GpuRenderer::available(&name) ) {
        return name;
    }
    return "no CUDA GPU (" + name + "): CPU rendering";
}

// One GPU renderer for the whole process (GPU memory is shared by all instances), used by one
// render at a time. It keeps the plan it was given last: switching between instances re-uploads.
static std::mutex gGpuMutex;
static ca::GpuRenderer* gGpu = 0;   // never deleted: the process may unload the plugin while CUDA shuts down
static unsigned long long gGpuPlanId = 0;
static bool gGpuBroken = false;
static std::atomic<unsigned long long> gNextPlanId(1);

// Everything the lens plan depends on
struct PlanKey
{
    std::vector<double> v;
    std::string zernike;
    bool operator==(const PlanKey& o) const { return v == o.v && zernike == o.zernike; }
    bool operator!=(const PlanKey& o) const { return !(*this == o); }
};

struct CachedPlan
{
    PlanKey key;
    std::shared_ptr<ca::LensPlan> plan;
    unsigned long long id = 0;
    std::string warning;
};

// Real lens choice: built-in prescription or file, and its settings
struct RealSpec
{
    bool real = false;
    std::string lens;          // built-in id or file path
    bool file = false;
    ca::RealLensSettings set;
    std::string camera;        // empty: CIE observer; built-in id or file path
    bool cameraFile = false;
};


// ----------------------------------------------------------------------------- presets
// Each preset first resets the optical parameters below, then sets its own values. A preset with
// a real lens also chooses the focal length that puts the image corner just inside the lens's
// design field (for the current filmback and image shape).
static const char* const kPresetResetParams[] = {
    kParamLensModel, kParamLens, kParamFocusDistance, kParamFNumber, kParamFocal, kParamBlades, kParamRoundness,
    kParamRotation, kParamPupilAspect, kParamVignetting, kParamVignettingStart, kParamNatural, kParamSpherical,
    kParamComa, kParamAstig, kParamCurvature, kParamDistortion, kParamDefocus, kParamAxialCA, kParamLateralCA,
    kParamSpheroChrom, kParamZernike, kParamAmount, kParamDistortionAmount, kParamLateralAmount, 0
};
struct PresetValue { const char* param; double value; };
struct Preset
{
    const char* label;
    const char* hint;
    const char* lens;            // built-in real lens id, or 0 (parametric)
    PresetValue values[12];      // ends with { 0, 0 }
};
static const Preset kPresets[] = {
    { "Diffraction Limited f/11", "A perfect lens: only the Airy disc and its rings remain.", 0,
      { { kParamFNumber, 11. }, { 0, 0. } } },
    { "Vintage (coma, astigmatism)", "Coma and astigmatism towards the corners, lateral colour, barrel distortion, "
      "cat's-eye vignetting.", 0,
      { { kParamFNumber, 2. }, { kParamSpherical, 6. }, { kParamComa, 20. }, { kParamAstig, 12. }, { kParamCurvature, 4. },
        { kParamDistortion, -2. }, { kParamLateralCA, 3. }, { kParamAxialCA, 2. }, { kParamBlades, 6. },
        { kParamRoundness, 0.5 }, { kParamVignetting, 0.6 }, { 0, 0. } } },
    { "Soft Wide Open f/1.4", "Spherical aberration and spherochromatism: soft glow, purple and green fringes.", 0,
      { { kParamFNumber, 1.4 }, { kParamSpherical, 6. }, { kParamDefocus, -6. }, { kParamSpheroChrom, 1.5 },
        { kParamAxialCA, 1.5 }, { 0, 0. } } },
    { "Diffraction Stars f/16", "Stopped down, 7 straight blades: 14-point diffraction stars on highlights.", 0,
      { { kParamFNumber, 16. }, { kParamBlades, 7. }, { kParamRoundness, 0. }, { kParamRotation, 10. }, { 0, 0. } } },
    { "Anamorphic Oval Bokeh", "Pupil twice as tall as wide, slightly out of focus: vertical oval bokeh.", 0,
      { { kParamFNumber, 2.8 }, { kParamPupilAspect, 2. }, { kParamDefocus, 6. }, { kParamAstig, 3. }, { kParamSpherical, 1. },
        { kParamAxialCA, 2. }, { 0, 0. } } },
    { "Real: Double Gauss f/2", "Modern Planar-type standard lens, ray traced.", "double_gauss_f2",
      { { kParamFNumber, 2. }, { 0, 0. } } },
    { "Real: Petzval f/1.4", "Portrait lens: sharp centre, swirling corners.", "petzval_f14",
      { { kParamFNumber, 1.4 }, { 0, 0. } } },
    { "Real: Cooke Triplet f/5", "Classic three-element lens.", "cooke_triplet_f5",
      { { kParamFNumber, 5. }, { 0, 0. } } },
    { "Real: Wide Angle f/4", "Retrofocus wide angle: strong barrel distortion, lateral colour, corner falloff.", "wide_angle_100",
      { { kParamFNumber, 4. }, { 0, 0. } } },
    { "Real: Telephoto f/5.6", "Telephoto lens.", "telephoto_f56",
      { { kParamFNumber, 5.6 }, { 0, 0. } } },
};
static const int kPresetCount = (int)( sizeof(kPresets) / sizeof(kPresets[0]) );


// ----------------------------------------------------------------------------- preset files
// Text files "key = value" (the parameter files of carender), read once when the host loads the
// plugin, from:
//   - the presets folder of the bundle (Contents/Resources/presets, examples shipped with the plugin);
//   - the user folder: %APPDATA%\ChromaAberrations_OFX\presets (Windows),
//     ~/.config/ChromaAberrations_OFX/presets (Linux), ~/Library/Application Support/... (macOS);
//   - the folders listed in the environment variable CA_PRESET_PATH (separated by ';', or ':' outside Windows).
// Keys: name, hint, lens (built-in id or .lens file, relative to the preset file), focal,
// focus_distance (m), fnumber, w040 w131 w222 w220 w311 w020, axial_ca, lateral_ca,
// spherochromatism, zernike, blades, roundness, rotation, aspect, vignetting, vignetting_start,
// natural_vignetting, aberration_gain, distortion_gain, lateral_gain, camera (built-in id or file).
// Other keys (render settings of carender: kernel, grid_x, wavelengths, output...) are ignored.
struct FilePreset
{
    std::string label, hint, path, dir;
    std::map<std::string, std::string> keys;
};

static std::string presetTrim(const std::string& s)
{
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

static std::string moduleDir()
{
#ifdef _WIN32
    HMODULE self = 0;
    wchar_t path[MAX_PATH * 2] = { 0 };
    if ( GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)&moduleDir, &self) && GetModuleFileNameW(self, path, MAX_PATH * 2) ) {
        char utf8[MAX_PATH * 6] = { 0 };
        WideCharToMultiByte(CP_UTF8, 0, path, -1, utf8, sizeof(utf8), 0, 0);
        std::string d(utf8);
        const size_t slash = d.find_last_of("\\/");
        return slash == std::string::npos ? std::string() : d.substr(0, slash);
    }
    return std::string();
#else
    Dl_info info;
    if ( dladdr( (void*)&moduleDir, &info ) && info.dli_fname ) {
        std::string d(info.dli_fname);
        const size_t slash = d.find_last_of('/');
        return slash == std::string::npos ? std::string(".") : d.substr(0, slash);
    }
    return std::string();
#endif
}

static std::vector<std::string> presetFolders()
{
    std::vector<std::string> dirs;
    const std::string mod = moduleDir();
    if ( !mod.empty() ) {
        dirs.push_back(mod + "/../Resources/presets");
    }
#ifdef _WIN32
    if ( const char* app = std::getenv("APPDATA") ) dirs.push_back(std::string(app) + "\\ChromaAberrations_OFX\\presets");
    const char sep = ';';
#else
    if ( const char* home = std::getenv("HOME") ) {
# ifdef __APPLE__
        dirs.push_back(std::string(home) + "/Library/Application Support/ChromaAberrations_OFX/presets");
# else
        const char* xdg = std::getenv("XDG_CONFIG_HOME");
        dirs.push_back( (xdg && *xdg ? std::string(xdg) : std::string(home) + "/.config") + "/ChromaAberrations_OFX/presets" );
# endif
    }
    const char sep = ':';
#endif
    if ( const char* env = std::getenv("CA_PRESET_PATH") ) {
        std::string e(env), cur;
        for (char c : e + std::string(1, sep)) {
            if ( (c == sep) || (c == ';') ) {
                if ( !presetTrim(cur).empty() ) dirs.push_back( presetTrim(cur) );
                cur.clear();
            } else {
                cur += c;
            }
        }
    }
    return dirs;
}

static std::vector<std::string> listTxtFiles(const std::string& dir)
{
    std::vector<std::string> files;
#ifdef _WIN32
    std::wstring wdir;
    {
        const int n = MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, 0, 0);
        std::vector<wchar_t> w(n > 0 ? n : 1);
        MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, w.data(), n);
        wdir = w.data();
    }
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW( (wdir + L"\\*.txt").c_str(), &fd );
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if ( !(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {
                char utf8[MAX_PATH * 6] = { 0 };
                WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, utf8, sizeof(utf8), 0, 0);
                files.push_back(dir + "\\" + utf8);
            }
        } while ( FindNextFileW(h, &fd) );
        FindClose(h);
    }
#else
    if ( DIR* d = opendir( dir.c_str() ) ) {
        while ( struct dirent* e = readdir(d) ) {
            const std::string n(e->d_name);
            if ( (n.size() > 4) && (n.compare(n.size() - 4, 4, ".txt") == 0) && (n[0] != '.') ) files.push_back(dir + "/" + n);
        }
        closedir(d);
    }
#endif
    std::sort(files.begin(), files.end());
    return files;
}

static const std::vector<FilePreset>& filePresets()
{
    static std::vector<FilePreset> presets;
    static std::once_flag once;
    std::call_once(once, []() {
        for (const std::string& dir : presetFolders()) {
            for (const std::string& path : listTxtFiles(dir)) {
#ifdef _WIN32
                std::ifstream f;
                {
                    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, 0, 0);
                    std::vector<wchar_t> w(n > 0 ? n : 1);
                    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
                    f.open( w.data() );
                }
#else
                std::ifstream f( path.c_str() );
#endif
                if (!f) continue;
                FilePreset fp;
                fp.path = path;
                const size_t slash = path.find_last_of("\\/");
                fp.dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash);
                std::string line;
                int n = 0;
                bool ok = true;
                while ( std::getline(f, line) ) {
                    ++n;
                    if ( (n == 1) && (line.size() >= 3) && ( (unsigned char)line[0] == 0xEF ) ) line = line.substr(3);   // UTF-8 BOM
                    std::string l = line;
                    const size_t eq = l.find('=');
                    std::string key = eq == std::string::npos ? presetTrim( l.substr( 0, l.find('#') ) ) : presetTrim( l.substr(0, eq) );
                    if (eq == std::string::npos) {
                        if ( !key.empty() ) {
                            std::fprintf(stderr, "ChromaAberrations_OFX: preset %s:%d: expected key = value\n", path.c_str(), n);
                            ok = false;
                        }
                        continue;
                    }
                    if ( key.empty() || (key[0] == '#') ) continue;
                    for (char& c : key) c = (char)std::tolower( (unsigned char)c );
                    std::string value = l.substr(eq + 1);
                    // comments: '#' after the value (names and hints may not contain '#')
                    value = presetTrim( value.substr( 0, value.find('#') ) );
                    if ( (value.size() >= 2) && (value.front() == '"') && (value.back() == '"') ) value = value.substr(1, value.size() - 2);
                    fp.keys[key] = value;
                }
                if (!ok) continue;
                std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
                base = base.substr(0, base.size() - 4);
                fp.label = fp.keys.count("name") && !fp.keys["name"].empty() ? fp.keys["name"] : base;
                fp.hint = fp.keys.count("hint") ? fp.keys["hint"] : std::string("Preset file ") + path;
                bool dup = false;
                for (const FilePreset& o : presets) dup = dup || (o.label == fp.label);
                if (dup) fp.label += " (" + base + ")";
                presets.push_back(fp);
            }
        }
    });
    return presets;
}

// ----------------------------------------------------------------------------- plugin

class ChromaAberrationsPlugin : public ImageEffect
{
public:
    explicit ChromaAberrationsPlugin(OfxImageEffectHandle handle)
        : ImageEffect(handle)
    {
        _dst = fetchClip(kOfxImageEffectOutputClipName);
        _src = getContext() == eContextGenerator ? 0 : fetchClip(kOfxImageEffectSimpleSourceClipName);
        _mask = (getContext() == eContextGeneral) ? fetchClip(kClipMask) : 0;
        _lensModel = fetchChoiceParam(kParamLensModel);
        _lens = fetchChoiceParam(kParamLens);
        _lensFile = fetchStringParam(kParamLensFile);
        _focusDistance = fetchDoubleParam(kParamFocusDistance);
        _distortionAmount = fetchDoubleParam(kParamDistortionAmount);
        _lateralAmount = fetchDoubleParam(kParamLateralAmount);
        _fNumber = fetchDoubleParam(kParamFNumber);
        _focal = fetchDoubleParam(kParamFocal);
        _filmback = fetchDoubleParam(kParamFilmback);
        _center = fetchDouble2DParam(kParamCenter);
        _centerOnImage = fetchBooleanParam(kParamCenterOnImage);
        _blades = fetchIntParam(kParamBlades);
        _roundness = fetchDoubleParam(kParamRoundness);
        _rotation = fetchDoubleParam(kParamRotation);
        _pupilAspect = fetchDoubleParam(kParamPupilAspect);
        _vignetting = fetchDoubleParam(kParamVignetting);
        _vignettingStart = fetchDoubleParam(kParamVignettingStart);
        _natural = fetchDoubleParam(kParamNatural);
        _spherical = fetchDoubleParam(kParamSpherical);
        _coma = fetchDoubleParam(kParamComa);
        _astig = fetchDoubleParam(kParamAstig);
        _curvature = fetchDoubleParam(kParamCurvature);
        _distortion = fetchDoubleParam(kParamDistortion);
        _defocus = fetchDoubleParam(kParamDefocus);
        _axialCA = fetchDoubleParam(kParamAxialCA);
        _lateralCA = fetchDoubleParam(kParamLateralCA);
        _spheroChrom = fetchDoubleParam(kParamSpheroChrom);
        _zernike = fetchStringParam(kParamZernike);
        _amount = fetchDoubleParam(kParamAmount);
        _colorSpace = fetchChoiceParam(kParamColorSpace);
        _camera = fetchChoiceParam(kParamCamera);
        _cameraFile = fetchStringParam(kParamCameraFile);
        _wlPreview = fetchIntParam(kParamWlPreview);
        _wlFinal = fetchIntParam(kParamWlFinal);
        _quality = fetchChoiceParam(kParamQuality);
        _gridPreview = fetchInt2DParam(kParamGridPreview);
        _gridFinal = fetchInt2DParam(kParamGridFinal);
        _maxPsf = fetchIntParam(kParamMaxPsf);
        _autoPsf = fetchBooleanParam(kParamAutoPsf);
        _gridMode = fetchChoiceParam(kParamGridMode);
        _gridTolerance = fetchDoubleParam(kParamGridTolerance);
        _useGPU = fetchBooleanParam(kParamUseGPU);
        _view = fetchChoiceParam(kParamView);
        _psfZoom = fetchIntParam(kParamPsfZoom);
        _psfBackground = fetchBooleanParam(kParamPsfBackground);
        _blur = fetchBooleanParam(kParamBlur);
        _distort = fetchBooleanParam(kParamDistort);
        _illum = fetchBooleanParam(kParamIllum);
        _mix = fetchDoubleParam(kParamMix);
        _maskInvert = fetchBooleanParam(kParamMaskInvert);
        updateEnabled();
    }

private:
    virtual void render(const RenderArguments& args) override final;
    virtual bool isIdentity(const IsIdentityArguments& args, Clip*& identityClip, double& identityTime
#ifdef OFX_EXTENSIONS_NUKE
                            , int& view, std::string& plane
#endif
                            ) override final;
    virtual bool getRegionOfDefinition(const RegionOfDefinitionArguments& args, OfxRectD& rod) override final;
    virtual void getRegionsOfInterest(const RegionsOfInterestArguments& args, RegionOfInterestSetter& rois) override final;
    virtual void changedParam(const InstanceChangedArgs& args, const std::string& paramName) override final;
    void applyPreset(int index, double time);
    void applyFilePreset(int index, double time);
    void resetOptics(bool parametric);
    void fitFocalToLens(const ca::LensPrescription& lp, double time);

    void updateEnabled();
    bool lensFromParams(double time, int width, int height, double par, const OfxRectI& bounds, const OfxPointD& rs,
                        bool interactive, ca::WavefrontParams& w, ca::ApertureParams& a, ca::ImageGeometry& g,
                        ca::RenderSettings& r, RealSpec& real, PlanKey& key);
    std::shared_ptr<ca::LensPlan> getPlan(const PlanKey& key, const ca::WavefrontParams& w, const ca::ApertureParams& a,
                                          const ca::ImageGeometry& g, const ca::RenderSettings& r, const RealSpec& real,
                                          unsigned long long& id, std::string& error, std::string& warning, bool& cached);

    Clip* _dst;
    Clip* _src;
    Clip* _mask;
    ChoiceParam *_lensModel, *_lens;
    StringParam* _lensFile;
    DoubleParam *_focusDistance, *_distortionAmount, *_lateralAmount;
    DoubleParam *_fNumber, *_focal, *_filmback;
    Double2DParam* _center;
    BooleanParam* _centerOnImage;
    IntParam* _blades;
    DoubleParam *_roundness, *_rotation, *_pupilAspect, *_vignetting, *_vignettingStart, *_natural;
    DoubleParam *_spherical, *_coma, *_astig, *_curvature, *_distortion, *_defocus, *_axialCA, *_lateralCA, *_spheroChrom;
    StringParam* _zernike;
    DoubleParam* _amount;
    ChoiceParam* _colorSpace;
    ChoiceParam* _camera;
    StringParam* _cameraFile;
    IntParam *_wlPreview, *_wlFinal;
    ChoiceParam* _quality;
    Int2DParam *_gridPreview, *_gridFinal;
    IntParam* _maxPsf;
    BooleanParam* _autoPsf;
    ChoiceParam* _gridMode;
    DoubleParam* _gridTolerance;
    BooleanParam* _useGPU;
    ChoiceParam* _view;
    IntParam* _psfZoom;
    BooleanParam* _psfBackground;
    BooleanParam *_blur, *_distort, *_illum;
    DoubleParam* _mix;
    BooleanParam* _maskInvert;

    std::mutex _planMutex;
    CachedPlan _cache;
};

void ChromaAberrationsPlugin::updateEnabled()
{
    int q = 0, v = 0;
    _quality->getValue(q);
    _view->getValue(v);
    int gm = 0;
    _gridMode->getValue(gm);
    _wlPreview->setEnabled(q != eQualityFinal);
    _gridPreview->setEnabled(true);   // also the start of the automatic Final grid
    _wlFinal->setEnabled(q != eQualityPreview);
    _gridMode->setEnabled(q != eQualityPreview);
    _gridFinal->setEnabled( (q != eQualityPreview) && (gm == 1) );
    _gridTolerance->setEnabled( (q != eQualityPreview) && (gm == 0) );
    _psfZoom->setEnabled(v == eViewPsfMap);
    _psfBackground->setEnabled(v == eViewPsfMap);
    _vignettingStart->setEnabled(_vignetting->getValue() > 0.);
    _center->setEnabled( !_centerOnImage->getValue() );
    int lm = 0, li = 0;
    _lensModel->getValue(lm);
    _lens->getValue(li);
    const bool real = lm == eLensReal;
    _lens->setEnabled(real);
    _lensFile->setEnabled( real && (li >= ca::builtinLensCount()) );
    int cam = 0;
    _camera->getValue(cam);
    _cameraFile->setEnabled(cam == ca::builtinCameraCount() + 1);
    _focusDistance->setEnabled(real);
    _distortionAmount->setEnabled(real);
    _lateralAmount->setEnabled(real);
    _pupilAspect->setEnabled(!real);
    _vignetting->setEnabled(!real);
    _vignettingStart->setEnabled( !real && (_vignetting->getValue() > 0.) );
    DoubleParam* para[] = { _spherical, _coma, _astig, _curvature, _distortion, _axialCA, _lateralCA, _spheroChrom };
    for (DoubleParam* p : para) {
        p->setEnabled(!real);
    }
    _zernike->setEnabled(!real);
}

void ChromaAberrationsPlugin::changedParam(const InstanceChangedArgs& args, const std::string& paramName)
{
    if ( (paramName == kParamPreset) && (args.reason == eChangeUserEdit) ) {
        int i = 0;
        fetchChoiceParam(kParamPreset)->getValue(i);
        if (i > 0) {
            applyPreset(i - 1, args.time);
            updateEnabled();
        }
        return;
    }
    if ( (paramName == kParamQuality) || (paramName == kParamView) || (paramName == kParamVignetting) ||
         (paramName == kParamCenterOnImage) || (paramName == kParamLensModel) || (paramName == kParamLens) ||
         (paramName == kParamGridMode) || (paramName == kParamCamera) ) {
        updateEnabled();
    }
}

// optical parameters back to their defaults; parametric: no aberrations at all
void ChromaAberrationsPlugin::resetOptics(bool parametric)
{
    for (int k = 0; kPresetResetParams[k]; ++k) {
        Param* p = getParam(kPresetResetParams[k]);
        if (!p) continue;
        switch ( p->getType() ) {
        case eDoubleParam: static_cast<DoubleParam*>(p)->resetToDefault(); break;
        case eIntParam: static_cast<IntParam*>(p)->resetToDefault(); break;
        case eChoiceParam: static_cast<ChoiceParam*>(p)->resetToDefault(); break;
        case eBooleanParam: static_cast<BooleanParam*>(p)->resetToDefault(); break;
        case eStringParam: static_cast<StringParam*>(p)->resetToDefault(); break;
        default: break;
        }
    }
    if (parametric) {
        // parametric presets start from a lens without aberrations (the defaults have a few)
        const char* const aberrations[] = { kParamSpherical, kParamComa, kParamAstig, kParamCurvature, kParamDistortion,
                                            kParamDefocus, kParamAxialCA, kParamLateralCA, kParamSpheroChrom };
        for (const char* n : aberrations) fetchDoubleParam(n)->setValue(0.);
    }
}

// focal length that puts the image corner at 95 % of the lens's design field
void ChromaAberrationsPlugin::fitFocalToLens(const ca::LensPrescription& lp, double time)
{
    if (lp.designHalfFieldDeg <= 0.) {
        return;
    }
    double aspect = 16. / 9.;
    if ( _src && _src->isConnected() ) {
        const OfxRectD rod = _src->getRegionOfDefinition(time);
        const double par = _src->getPixelAspectRatio();
        if ( (rod.x2 > rod.x1) && (rod.y2 > rod.y1) ) aspect = (rod.x2 - rod.x1) * par / (rod.y2 - rod.y1);
    }
    const double halfDiag = 0.5 * _filmback->getValue() * std::sqrt(1. + 1. / (aspect * aspect));
    const double f = halfDiag / std::tan(0.95 * lp.designHalfFieldDeg * M_PI / 180.);
    _focal->setValue( std::max(1., std::round(f)) );
}

void ChromaAberrationsPlugin::applyPreset(int index, double time)
{
    if (index >= kPresetCount) {
        applyFilePreset(index - kPresetCount, time);
        return;
    }
    if (index < 0) {
        return;
    }
    const Preset& pr = kPresets[index];
    beginEditBlock(std::string("Preset ") + pr.label);
    resetOptics(!pr.lens);
    for (int k = 0; pr.values[k].param; ++k) {
        Param* p = getParam(pr.values[k].param);
        if (!p) continue;
        const double v = pr.values[k].value;
        switch ( p->getType() ) {
        case eDoubleParam: static_cast<DoubleParam*>(p)->setValue(v); break;
        case eIntParam: static_cast<IntParam*>(p)->setValue( (int)std::lround(v) ); break;
        case eChoiceParam: static_cast<ChoiceParam*>(p)->setValue( (int)std::lround(v) ); break;
        case eBooleanParam: static_cast<BooleanParam*>(p)->setValue(v != 0.); break;
        default: break;
        }
    }
    if (pr.lens) {
        _lensModel->setValue(eLensReal);
        for (int i = 0; i < ca::builtinLensCount(); ++i) {
            if (std::string(ca::builtinLensId(i)) == pr.lens) {
                _lens->setValue(i);
                ca::LensPrescription lp;
                if ( ca::loadBuiltinLens(pr.lens, lp, 0) ) fitFocalToLens(lp, time);
                break;
            }
        }
    }
    endEditBlock();
}

static std::string resolvePresetPath(const std::string& dir, const std::string& f)
{
    const bool absolute = !f.empty() && ( (f[0] == '/') || (f[0] == '\\') || ( (f.size() > 1) && (f[1] == ':') ) );
    return absolute ? f : dir + "/" + f;
}

// Preset file: same keys, units and defaults as carender, so a file gives the same lens in both
void ChromaAberrationsPlugin::applyFilePreset(int index, double time)
{
    const std::vector<FilePreset>& list = filePresets();
    if ( (index < 0) || (index >= (int)list.size()) ) {
        return;
    }
    const FilePreset& fp = list[index];
    const std::map<std::string, std::string>& k = fp.keys;
    auto has = [&](const char* key) { return k.count(key) && !k.at(key).empty(); };
    auto num = [&](const char* key, double def) { return has(key) ? std::atof( k.at(key).c_str() ) : def; };
    std::vector<std::string> problems;
    const bool real = has("lens");
    beginEditBlock("Preset " + fp.label);
    resetOptics(!real);
    if (real) {
        _lensModel->setValue(eLensReal);
        const std::string id = k.at("lens");
        ca::LensPrescription lp;
        bool found = false;
        for (int i = 0; i < ca::builtinLensCount(); ++i) {
            if (id == ca::builtinLensId(i)) {
                _lens->setValue(i);
                found = ca::loadBuiltinLens(id, lp, 0);
            }
        }
        if (!found) {
            const std::string path = resolvePresetPath(fp.dir, id);
            _lens->setValue( ca::builtinLensCount() );   // Custom File
            _lensFile->setValue(path);
            std::string err;
            found = ca::loadLensFile(path, lp, &err);
            if (!found) problems.push_back("lens " + id + ": " + err);
        }
        if ( has("focal") ) {
            _focal->setValue( num("focal", 50.) );
        } else if (found) {
            fitFocalToLens(lp, time);
        }
        if ( has("fnumber") ) _fNumber->setValue( num("fnumber", 2.8) );
        else if (found) _fNumber->setValue(lp.designFNumber);
        if ( has("focus_distance") ) _focusDistance->setValue( num("focus_distance", 0.) );
        if ( has("aberration_gain") ) _amount->setValue( num("aberration_gain", 1.) );
        if ( has("distortion_gain") ) _distortionAmount->setValue( num("distortion_gain", 1.) );
        if ( has("lateral_gain") ) _lateralAmount->setValue( num("lateral_gain", 1.) );
    } else {
        _fNumber->setValue( num("fnumber", 2.8) );
        _spherical->setValue( num("w040", 0.) );
        _coma->setValue( num("w131", 0.) );
        _astig->setValue( num("w222", 0.) );
        _curvature->setValue( num("w220", 0.) );
        _distortion->setValue( num("w311", 0.) );
        _defocus->setValue( num("w020", 0.) );
        _axialCA->setValue( num("axial_ca", 0.) );
        _lateralCA->setValue( num("lateral_ca", 0.) );
        _spheroChrom->setValue( num("spherochromatism", 0.) );
        if ( has("zernike") ) _zernike->setValue( k.at("zernike") );
        if ( has("focal") ) _focal->setValue( num("focal", 35.) );
    }
    // diaphragm and vignetting (both lens models)
    _blades->setValue( (int)num("blades", 0.) );
    _roundness->setValue( num("roundness", 0.) );
    _rotation->setValue( num("rotation", 0.) );
    _pupilAspect->setValue( num("aspect", 1.) );
    _vignetting->setValue( num("vignetting", 0.) );
    _vignettingStart->setValue( num("vignetting_start", 0.5) );
    if ( has("natural_vignetting") ) _natural->setValue( num("natural_vignetting", 1.) );
    // camera: only when the preset names one (otherwise it belongs to the plate, not the lens)
    if ( has("camera") ) {
        const std::string c = k.at("camera");
        int idx = -1;
        if ( (c == "cie") || (c == "none") ) idx = 0;
        for (int i = 0; i < ca::builtinCameraCount(); ++i) {
            if (c == ca::builtinCameraId(i)) idx = i + 1;
        }
        if (idx < 0) {
            idx = ca::builtinCameraCount() + 1;   // Custom File
            _cameraFile->setValue( resolvePresetPath(fp.dir, c) );
        }
        _camera->setValue(idx);
    }
    endEditBlock();
    if ( !problems.empty() ) {
        std::string m = "Preset " + fp.label + " (" + fp.path + "): ";
        for (const std::string& pb : problems) m += pb + ". ";
        sendMessage(Message::eMessageError, "", m);
    }
}

bool ChromaAberrationsPlugin::lensFromParams(double t, int width, int height, double par, const OfxRectI& bounds,
                                             const OfxPointD& rs, bool interactive, ca::WavefrontParams& w,
                                             ca::ApertureParams& a, ca::ImageGeometry& g, ca::RenderSettings& r,
                                             RealSpec& real, PlanKey& key)
{
    const double amount = _amount->getValueAtTime(t);
    w.fNumber = std::max(0.5, _fNumber->getValueAtTime(t));
    w.lambda0Um = 0.55;
    w.W040 = amount * _spherical->getValueAtTime(t);
    w.W131 = amount * _coma->getValueAtTime(t);
    w.W222 = amount * _astig->getValueAtTime(t);
    w.W220 = amount * _curvature->getValueAtTime(t);
    w.W311 = amount * _distortion->getValueAtTime(t);
    w.W020 = amount * _defocus->getValueAtTime(t);
    w.axialCA = amount * _axialCA->getValueAtTime(t);
    w.lateralCA = amount * _lateralCA->getValueAtTime(t);
    w.spherochromatism = amount * _spheroChrom->getValueAtTime(t);
    std::string z;
    _zernike->getValueAtTime(t, z);
    w.zernike.clear();
    {
        std::string s = z;
        std::replace(s.begin(), s.end(), ';', ',');
        std::replace(s.begin(), s.end(), ' ', ',');
        std::stringstream ss(s);
        std::string item;
        while ( std::getline(ss, item, ',') ) {
            const size_t c = item.find(':');
            if (c == std::string::npos) {
                continue;
            }
            const int j = std::atoi( item.substr(0, c).c_str() );
            if (j >= 4) {
                w.zernike.push_back( std::make_pair( j, amount * std::atof( item.substr(c + 1).c_str() ) ) );
            }
        }
    }
    a.blades = _blades->getValueAtTime(t);
    a.roundness = _roundness->getValueAtTime(t);
    a.rotationDeg = _rotation->getValueAtTime(t);
    a.aspect = std::max(0.1, _pupilAspect->getValueAtTime(t));
    a.vignetting = _vignetting->getValueAtTime(t);
    a.vignettingStart = _vignettingStart->getValueAtTime(t);

    g.width = width;
    g.height = height;
    g.filmbackWidthMm = std::max(0.1, _filmback->getValueAtTime(t));
    g.pixelAspect = par > 0. ? par : 1.;
    // optical centre: centre of the image, or the position given (canonical coordinates -> fraction)
    if ( _centerOnImage->getValueAtTime(t) ) {
        g.centerX = g.centerY = 0.5;
    } else {
        double cx, cy;
        _center->getValueAtTime(t, cx, cy);
        const double px = cx * rs.x / g.pixelAspect, py = cy * rs.y;
        g.centerX = (px - bounds.x1) / std::max(1, width);
        g.centerY = (py - bounds.y1) / std::max(1, height);
    }

    int q = 0, cs = 0;
    _quality->getValueAtTime(t, q);
    _colorSpace->getValueAtTime(t, cs);
    int gx = 7, gy = 5;
    const bool final = (q == eQualityFinal) || ( (q == eQualityAuto) && !interactive );
    int gridMode = 0;
    _gridMode->getValueAtTime(t, gridMode);
    if (final) {
        r.wavelengths = std::max(3, _wlFinal->getValueAtTime(t));
        if (gridMode == 0) {
            // automatic: refined from the Preview grid until the tolerance is met
            _gridPreview->getValueAtTime(t, gx, gy);
            r.autoGrid = true;
            r.gridTolerance = std::max(0.002, 0.01 * _gridTolerance->getValueAtTime(t));
        } else {
            _gridFinal->getValueAtTime(t, gx, gy);
        }
    } else {
        r.wavelengths = std::max(3, _wlPreview->getValueAtTime(t));
        _gridPreview->getValueAtTime(t, gx, gy);
    }
    r.gridX = std::max(1, gx);
    r.gridY = std::max(1, gy);
    // the PSF size is given at full resolution: it follows the render scale (proxy, viewer downscale)
    r.kernelSize = std::max( 9, (int)std::lround(_maxPsf->getValueAtTime(t) * rs.x) ) | 1;
    r.autoKernel = _autoPsf->getValueAtTime(t);
    r.colorSpace = cs == 1 ? ca::eColorSpaceRec709 : (cs == 2 ? ca::eColorSpaceRec2020 : ca::eColorSpaceACEScg);
    r.blur = _blur->getValueAtTime(t);
    r.distortion = _distort->getValueAtTime(t);
    r.vignetting = _illum->getValueAtTime(t);
    r.focalLengthMm = std::max(1., _focal->getValueAtTime(t));
    r.naturalVignetting = _natural->getValueAtTime(t);

    // real lens
    int lm = 0, li = 0;
    _lensModel->getValueAtTime(t, lm);
    _lens->getValueAtTime(t, li);
    real.real = lm == eLensReal;
    real.file = li >= ca::builtinLensCount();
    if (real.file) {
        _lensFile->getValueAtTime(t, real.lens);
    } else {
        real.lens = ca::builtinLensId(li);
    }
    real.set.focalMm = r.focalLengthMm;
    real.set.fNumber = w.fNumber;
    real.set.focusDistanceMm = std::max(0., _focusDistance->getValueAtTime(t)) * 1000.;
    real.set.blades = a.blades;
    real.set.roundness = a.roundness;
    real.set.rotationDeg = a.rotationDeg;
    real.set.aberrationGain = amount;
    real.set.distortionGain = _distortionAmount->getValueAtTime(t);
    real.set.lateralColourGain = _lateralAmount->getValueAtTime(t);
    real.set.defocusWaves = w.W020;
    int cam = 0;
    _camera->getValueAtTime(t, cam);
    real.cameraFile = cam == ca::builtinCameraCount() + 1;
    if (real.cameraFile) {
        _cameraFile->getValueAtTime(t, real.camera);
    } else if (cam >= 1) {
        real.camera = ca::builtinCameraId(cam - 1);
    }

    const double v[] = { (double)r.autoKernel, (double)r.autoGrid, r.gridTolerance, (double)real.real, real.set.focusDistanceMm, real.set.distortionGain, real.set.lateralColourGain,
                         w.fNumber, w.W040, w.W131, w.W222, w.W220, w.W311, w.W020, w.axialCA, w.lateralCA, w.spherochromatism,
                         (double)a.blades, a.roundness, a.rotationDeg, a.aspect, a.vignetting, a.vignettingStart,
                         (double)g.width, (double)g.height, g.filmbackWidthMm, g.pixelAspect, g.centerX, g.centerY,
                         (double)r.wavelengths, (double)r.gridX, (double)r.gridY, (double)r.kernelSize, (double)r.colorSpace,
                         (double)r.blur, (double)r.distortion, (double)r.vignetting, r.focalLengthMm, r.naturalVignetting };
    key.v.assign( v, v + sizeof(v) / sizeof(v[0]) );
    std::ostringstream zs;
    for (auto& p : w.zernike) {
        zs << p.first << ':' << p.second << ',';
    }
    key.zernike = zs.str() + "|" + (real.real ? real.lens : std::string()) + "|" + real.camera;
    return true;
}

std::shared_ptr<ca::LensPlan> ChromaAberrationsPlugin::getPlan(const PlanKey& key, const ca::WavefrontParams& w,
                                                               const ca::ApertureParams& a, const ca::ImageGeometry& g,
                                                               const ca::RenderSettings& r, const RealSpec& real,
                                                               unsigned long long& id, std::string& error, std::string& warning,
                                                               bool& cached)
{
    std::lock_guard<std::mutex> lock(_planMutex);
    cached = false;
    if (_cache.plan && (_cache.key == key)) {
        id = _cache.id;
        warning = _cache.warning;
        cached = true;
        return _cache.plan;
    }
    // camera sensitivities
    ca::RenderSettings rs = r;
    if ( !real.camera.empty() ) {
        std::shared_ptr<ca::CameraSensitivity> cam = std::make_shared<ca::CameraSensitivity>();
        std::string err;
        if ( real.cameraFile ? !ca::loadCameraFile(real.camera, *cam, &err) : !ca::builtinCamera(real.camera, *cam) ) {
            error = "camera: " + (err.empty() ? real.camera : err);
            return std::shared_ptr<ca::LensPlan>();
        }
        rs.camera = cam;
    } else if (real.cameraFile) {
        error = "choose a camera sensitivity file";
        return std::shared_ptr<ca::LensPlan>();
    }
    // the lens model
    std::unique_ptr<ca::LensModel> model;
    warning.clear();
    if (real.real) {
        ca::LensPrescription pr;
        std::string err;
        const bool ok = real.file ? ( !real.lens.empty() && ca::loadLensFile(real.lens, pr, &err) ) : ca::loadBuiltinLens(real.lens, pr, &err);
        if (!ok) {
            error = real.lens.empty() ? std::string("choose a lens file") : err;
            return std::shared_ptr<ca::LensPlan>();
        }
        ca::RealLens L;
        if ( !L.setup(pr, real.set, &err) ) {
            error = pr.name + ": " + err;
            return std::shared_ptr<ca::LensPlan>();
        }
        std::ostringstream wm;
        if (real.set.fNumber < pr.designFNumber - 1e-6) {
            wm << pr.name << " opens to f/" << pr.designFNumber << " only. ";
        }
        ca::RealLensModel* m = new ca::RealLensModel(L);
        model.reset(m);
        // field of the image corner, against the design field (computed once the model is prepared)
        warning = wm.str();
    } else {
        model.reset( new ca::ParametricLensModel(w, a) );
    }
    // compute in a worker thread; this (render) thread reports the progress to the host
    std::shared_ptr<ca::LensPlan> plan = std::make_shared<ca::LensPlan>();
    std::atomic<int> done(0), total(1);
    std::atomic<bool> finished(false);
    bool built = false;
    std::string buildError;
    std::thread worker([&]() {
        built = ca::buildLensPlan(*model, g, rs, *plan, [&](int d, int tt) { done = d; total = tt; }, &buildError);
        finished = true;
    });
    progressStart("ChromaAberrations_OFX: computing the lens PSFs");
    while (!finished) {
        std::this_thread::sleep_for( std::chrono::milliseconds(50) );
        progressUpdate( (double)done / std::max(1, (int)total) );
    }
    worker.join();
    progressEnd();
    if (!built) {
        error = buildError;
        return std::shared_ptr<ca::LensPlan>();
    }
    if (real.real) {
        const ca::RealLensModel* m = static_cast<const ca::RealLensModel*>( model.get() );
        const double design = m->lens().prescription().designHalfFieldDeg;
        if ( m->lens().infiniteObject() && (m->imageHalfFieldDeg() > 1.05 * design) ) {
            std::ostringstream wm;
            wm.precision(3);
            wm << m->lens().prescription().name << ": the image corner is at " << m->imageHalfFieldDeg()
               << " deg, the lens was designed for " << design << " deg: the corners are vignetted "
               << "(use a longer focal length or a smaller filmback).";
            warning += wm.str();
        }
    }
    _cache.key = key;
    _cache.plan = plan;
    _cache.id = gNextPlanId++;
    _cache.warning = warning;
    id = _cache.id;
    return plan;
}

bool ChromaAberrationsPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    if ( !_src || !_src->isConnected() ) {
        return false;
    }
    rod = _src->getRegionOfDefinition(args.time);
    return true;
}

void ChromaAberrationsPlugin::getRegionsOfInterest(const RegionsOfInterestArguments& args, RegionOfInterestSetter& rois)
{
    // distortion and blur can bring light from anywhere in the image
    if ( _src && _src->isConnected() ) {
        rois.setRegionOfInterest( *_src, _src->getRegionOfDefinition(args.time) );
    }
}

bool ChromaAberrationsPlugin::isIdentity(const IsIdentityArguments& args, Clip*& identityClip, double& /*identityTime*/
#ifdef OFX_EXTENSIONS_NUKE
                                         , int& /*view*/, std::string& /*plane*/
#endif
                                         )
{
    int v = 0;
    _view->getValueAtTime(args.time, v);
    if (v != eViewImage) {
        return false;
    }
    const bool off = !_blur->getValueAtTime(args.time) && !_distort->getValueAtTime(args.time) && !_illum->getValueAtTime(args.time);
    if ( off || (_mix->getValueAtTime(args.time) <= 0.) ) {
        identityClip = _src;
        return true;
    }
    return false;
}

void ChromaAberrationsPlugin::render(const RenderArguments& args)
{
    typedef std::chrono::steady_clock Clock;
    const auto t0 = Clock::now();
    auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const double t = args.time;
    std::unique_ptr<Image> dst( _dst->fetchImage(t) );
    if ( !dst.get() ) {
        throwSuiteStatusException(kOfxStatFailed);
    }
    if (dst->getPixelDepth() != eBitDepthFloat) {
        throwSuiteStatusException(kOfxStatErrImageFormat);
    }
    std::unique_ptr<const Image> src( (_src && _src->isConnected()) ? _src->fetchImage(t) : 0 );
    std::unique_ptr<const Image> mask( (_mask && _mask->isConnected()) ? _mask->fetchImage(t) : 0 );
    const auto tFetch = Clock::now();   // the host may compute the upstream nodes while fetching
    const OfxRectI win = args.renderWindow;
    const int W = win.x2 - win.x1, H = win.y2 - win.y1;
    if ( (W <= 0) || (H <= 0) ) {
        return;
    }
    const int dstComps = dst->getPixelComponentCount();
    const int srcComps = src.get() ? src->getPixelComponentCount() : 0;
    const double par = src.get() ? src->getPixelAspectRatio() : dst->getPixelAspectRatio();
    const size_t n = (size_t)W * H;
    int view = 0;
    _view->getValueAtTime(t, view);

    // source rows into three planes (straight from the host's image: no intermediate copy)
    auto packSource = [&](float* r, float* g, float* b) {
        parallelFor(H, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const size_t o = (size_t)y * W;
                const float* p = src.get() && (srcComps >= 3) ? (const float*)src->getPixelAddress(win.x1, win.y1 + y) : 0;
                if (!p) {
                    std::fill(r + o, r + o + W, 0.f);
                    std::fill(g + o, g + o + W, 0.f);
                    std::fill(b + o, b + o + W, 0.f);
                    continue;
                }
                // pixels of the row are contiguous when the source covers the window
                const float* pe = (const float*)src->getPixelAddress(win.x2 - 1, win.y1 + y);
                if ( pe && (pe == p + (size_t)(W - 1) * srcComps) && (srcComps == 4) ) {
                    float* __restrict rr = r + o;
                    float* __restrict gg = g + o;
                    float* __restrict bb = b + o;
                    for (int x = 0; x < W; ++x) {   // constant stride: vectorised
                        rr[x] = p[4 * x];
                        gg[x] = p[4 * x + 1];
                        bb[x] = p[4 * x + 2];
                    }
                } else if ( pe && (pe == p + (size_t)(W - 1) * srcComps) ) {
                    for (int x = 0; x < W; ++x, p += srcComps) {
                        r[o + x] = p[0];
                        g[o + x] = p[1];
                        b[o + x] = p[2];
                    }
                } else {
                    for (int x = 0; x < W; ++x) {
                        const float* q = (const float*)src->getPixelAddress(win.x1 + x, win.y1 + y);
                        r[o + x] = q ? q[0] : 0.f;
                        g[o + x] = q ? q[1] : 0.f;
                        b[o + x] = q ? q[2] : 0.f;
                    }
                }
            }
        });
    };

    // ---- lens plan
    ca::WavefrontParams w;
    ca::ApertureParams a;
    ca::ImageGeometry g;
    ca::RenderSettings rset;
    PlanKey key;
    RealSpec real;
    lensFromParams(t, W, H, par, win, args.renderScale, args.interactiveRenderStatus, w, a, g, rset, real, key);
    unsigned long long planId = 0;
    std::string planError, planWarning;
    bool planCached = false;
    std::shared_ptr<ca::LensPlan> plan = getPlan(key, w, a, g, rset, real, planId, planError, planWarning, planCached);
    if (!plan) {
        setPersistentMessage(Message::eMessageError, "", "ChromaAberrations_OFX: " + planError);
        throwSuiteStatusException(kOfxStatFailed);
    }
    if ( planWarning.empty() ) {
        clearPersistentMessage();
    } else {
        setPersistentMessage(Message::eMessageWarning, "", planWarning);
    }
    if ( abort() ) {
        return;
    }
    const auto tPlan = Clock::now();

    // ---- effect
    ca::ImageRGB in, out;
    const float* outPlanes[3] = { 0, 0, 0 };
    bool done = false, onGpu = false;
    ca::RenderStats st;
    double packMs = 0.;
    std::unique_lock<std::mutex> gpuLock(gGpuMutex, std::defer_lock);
    if (view == eViewPsfMap) {
        const bool behind = _psfBackground->getValueAtTime(t);
        if (behind) {
            in.resize(W, H);
            packSource(in.c[0].data(), in.c[1].data(), in.c[2].data());
        }
        ca::makePsfMapView(*plan, behind ? &in : 0, _psfZoom->getValueAtTime(t), out);
        done = true;
    } else if (view == eViewTestChart) {
        ca::makeTestChart("chart", W, H, in);
    }
    if ( !done && _useGPU->getValueAtTime(t) ) {
        gpuLock.lock();
        if ( !gGpuBroken && ca::GpuRenderer::available() ) {
            if (!gGpu) {
                gGpu = new ca::GpuRenderer;
            }
            std::string err;
            bool ok = true;
            if (gGpuPlanId != planId) {
                ok = gGpu->setPlan(*plan, &err);
                gGpuPlanId = ok ? planId : 0;
            }
            float* hin = ok ? gGpu->inputBuffer(W, H) : 0;
            if (hin) {
                const auto tp = Clock::now();
                if (view == eViewTestChart) {
                    for (int c = 0; c < 3; ++c) std::memcpy(hin + c * n, in.c[c].data(), n * sizeof(float));
                } else {
                    packSource(hin, hin + n, hin + 2 * n);   // page-locked buffer, filled from the host's image
                }
                packMs = ms(tp, Clock::now());
                ok = gGpu->renderBuffers(W, H, &st, &err);
                if (ok) {
                    const float* o = gGpu->outputBuffer();
                    outPlanes[0] = o;
                    outPlanes[1] = o + n;
                    outPlanes[2] = o + 2 * n;
                    done = onGpu = true;   // the GPU lock stays held until the output is copied
                }
            } else if (ok) {
                ok = false;
                err = "cannot allocate the transfer buffers";
            }
            if (!ok) {
                std::fprintf(stderr, "ChromaAberrations_OFX: GPU render failed (%s), using the CPU\n", err.c_str());
                gGpuPlanId = 0;
            }
        }
        if (!done) {
            gpuLock.unlock();
        }
    }
    if (!done) {
        const auto tp = Clock::now();
        if (view == eViewImage) {
            in.resize(W, H);
            packSource(in.c[0].data(), in.c[1].data(), in.c[2].data());
        }
        packMs = ms(tp, Clock::now());
        ca::renderCpu(*plan, in, out, &st);
        done = true;
    }
    if (!outPlanes[0]) {
        outPlanes[0] = out.c[0].data();
        outPlanes[1] = out.c[1].data();
        outPlanes[2] = out.c[2].data();
    }
    const auto tEffect = Clock::now();

    // ---- output: mix and mask with the source (image view), alpha from the source
    const double mix = view == eViewImage ? _mix->getValueAtTime(t) : 1.;
    const bool invert = _maskInvert->getValueAtTime(t);
    const bool plain = (view != eViewImage) || ( (mix >= 1.) && !mask.get() && !invert );
    parallelFor(H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            float* d = (float*)dst->getPixelAddress(win.x1, win.y1 + y);
            if (!d) {
                continue;
            }
            const float* srow = src.get() ? (const float*)src->getPixelAddress(win.x1, win.y1 + y) : 0;
            const float* send = src.get() ? (const float*)src->getPixelAddress(win.x2 - 1, win.y1 + y) : 0;
            const bool contiguous = srow && send && (send == srow + (size_t)(W - 1) * srcComps);
            if ( plain && (dstComps == 4) ) {
                const float* dEnd = (const float*)dst->getPixelAddress(win.x2 - 1, win.y1 + y);
                if (dEnd == d + (size_t)(W - 1) * 4) {
                    // common case, constant strides: vectorised
                    const size_t o = (size_t)y * W;
                    const float* __restrict r = outPlanes[0] + o;
                    const float* __restrict g = outPlanes[1] + o;
                    const float* __restrict b = outPlanes[2] + o;
                    const bool srcAlpha = (view == eViewImage) && contiguous && (srcComps == 4);
                    for (int x = 0; x < W; ++x) {
                        d[4 * x] = r[x];
                        d[4 * x + 1] = g[x];
                        d[4 * x + 2] = b[x];
                        d[4 * x + 3] = srcAlpha ? srow[4 * x + 3] : 1.f;
                    }
                    continue;
                }
            }
            for (int x = 0; x < W; ++x, d += dstComps) {
                const size_t o = (size_t)y * W + x;
                const float* sp = contiguous ? srow + (size_t)x * srcComps
                                             : ( src.get() ? (const float*)src->getPixelAddress(win.x1 + x, win.y1 + y) : 0 );
                if (plain) {
                    for (int c = 0; c < std::min(3, dstComps); ++c) d[c] = outPlanes[c][o];
                } else {
                    double m = mix;
                    if (mask.get()) {
                        const float* mp = (const float*)mask->getPixelAddress(win.x1 + x, win.y1 + y);
                        double mv = mp ? mp[mask->getPixelComponentCount() - 1] : 0.;
                        if (invert) {
                            mv = 1. - mv;
                        }
                        m *= std::min(1., std::max(0., mv));
                    } else if (invert) {
                        m = 0.;
                    }
                    for (int c = 0; c < std::min(3, dstComps); ++c) {
                        const float e = outPlanes[c][o];
                        const float s = (sp && srcComps >= 3) ? sp[c] : 0.f;
                        d[c] = (float)(s + (e - s) * m);
                    }
                }
                if (dstComps == 4) {
                    d[3] = (view == eViewImage && sp && srcComps == 4) ? sp[3] : 1.f;
                }
            }
        }
    });
    if (gpuLock.owns_lock()) {
        gpuLock.unlock();
    }

    // ---- timing log: set the environment variable CA_PROFILE=1 before starting the host
    static const bool profile = std::getenv("CA_PROFILE") != 0;
    if (profile) {
        const auto tEnd = Clock::now();
        std::fprintf(stderr, "ChromaAberrations_OFX profile: frame %g, %dx%d, %s, %d wavelengths, grid %dx%d%s, PSF %d px | "
                     "input %.0f ms | lens plan %s %.0f ms (lens %.0f, PSFs %.0f, maps %.0f) | pack %.1f ms | %s %.1f ms",
                     t, W, H, view == eViewImage ? "image" : (view == eViewPsfMap ? "PSF map" : "test chart"),
                     plan->wavelengthCount(), plan->gridX, plan->gridY, plan->gridError >= 0. ? " (auto)" : "", plan->S,
                     ms(t0, tFetch), planCached ? "cached" : "computed", ms(tFetch, tPlan), plan->prepareSeconds * 1e3, plan->psfSeconds * 1e3,
                     plan->mapsSeconds * 1e3, packMs, onGpu ? "GPU" : "CPU", ms(tPlan, tEffect) - packMs);
        if (onGpu) {
            std::fprintf(stderr, " (transfers %.1f, compute %.1f, %d batch%s, PSF spectra %s)", st.transferSeconds * 1e3,
                         st.fftSeconds * 1e3, st.batches, st.batches > 1 ? "es" : "", st.kernelCache ? "kept" : "recomputed");
        }
        std::fprintf(stderr, " | output %.1f ms | total %.1f ms\n", ms(tEffect, tEnd), ms(t0, tEnd));
    }
}

// ----------------------------------------------------------------------------- factory

mDeclarePluginFactory(ChromaAberrationsPluginFactory, {}, {});

void ChromaAberrationsPluginFactory::describe(ImageEffectDescriptor& desc)
{
    desc.setLabel(kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);
    desc.addSupportedContext(eContextFilter);
    desc.addSupportedContext(eContextGeneral);
    desc.addSupportedBitDepth(eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(false);            // the whole image is needed (distortion, blur)
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(eRenderInstanceSafe);
    gIsNuke = getImageEffectHostDescription()->hostName.find("thefoundry.nuke") != std::string::npos;
}

static DoubleParamDescriptor* defineDouble(ImageEffectDescriptor& desc, PageParamDescriptor* page, const char* name,
                                           const char* label, const char* hint, double def, double lo, double hi,
                                           double dlo, double dhi)
{
    DoubleParamDescriptor* p = desc.defineDoubleParam(name);
    p->setLabel(label);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(lo, hi);
    p->setDisplayRange(dlo, dhi);
    p->setAnimates(true);
    if (page) {
        page->addChild(*p);
    }
    return p;
}

static IntParamDescriptor* defineInt(ImageEffectDescriptor& desc, PageParamDescriptor* page, const char* name,
                                     const char* label, const char* hint, int def, int lo, int hi)
{
    IntParamDescriptor* p = desc.defineIntParam(name);
    p->setLabel(label);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(lo, hi);
    p->setDisplayRange(lo, hi);
    p->setAnimates(false);
    if (page) {
        page->addChild(*p);
    }
    return p;
}

static BooleanParamDescriptor* defineBool(ImageEffectDescriptor& desc, PageParamDescriptor* page, const char* name,
                                          const char* label, const char* hint, bool def)
{
    BooleanParamDescriptor* p = desc.defineBooleanParam(name);
    p->setLabel(label);
    p->setHint(hint);
    p->setDefault(def);
    p->setAnimates(false);
    if (page) {
        page->addChild(*p);
    }
    return p;
}

void ChromaAberrationsPluginFactory::describeInContext(ImageEffectDescriptor& desc, ContextEnum context)
{
    // ---- clips
    ClipDescriptor* srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(false);
    srcClip->setIsMask(false);
    if (context == eContextGeneral) {
        ClipDescriptor* maskClip = desc.defineClip(kClipMask);
        maskClip->addSupportedComponent(ePixelComponentAlpha);
        maskClip->setTemporalClipAccess(false);
        maskClip->setOptional(true);
        maskClip->setSupportsTiles(false);
        maskClip->setIsMask(true);
    }
    ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentRGB);
    dstClip->setSupportsTiles(false);

    // ---- Lens
    PageParamDescriptor* page = desc.definePageParam("Lens");
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamPreset);
        p->setLabel("Preset");
        std::string hint = "Sets the lens parameters (Lens and Aberrations tabs) to a ready-made lens; they can then be "
                           "edited freely. The filmback, optical centre, spectral and render settings are kept.\n";
        p->appendOption("Choose...");
        for (int i = 0; i < kPresetCount; ++i) {
            p->appendOption(kPresets[i].label);
            hint += std::string("\n") + kPresets[i].label + ": " + kPresets[i].hint;
        }
        // preset files (read once, when the host loads the plugin: restart the host after adding one)
        for (const FilePreset& fp : filePresets()) {
            p->appendOption(fp.label);
            hint += "\n" + fp.label + ": " + fp.hint;
        }
        hint += "\n\nMore presets: text files (.txt) in the presets folder of the plugin bundle, in "
#ifdef _WIN32
                "%APPDATA%\\ChromaAberrations_OFX\\presets"
#else
                "~/.config/ChromaAberrations_OFX/presets"
#endif
                " or in the folders of the CA_PRESET_PATH environment variable. Restart the host after adding one.";
        p->setHint(hint);
        p->setDefault(0);
        p->setAnimates(false);
        p->setEvaluateOnChange(false);
        p->setLayoutHint(eLayoutHintDivider);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamLensModel);
        p->setLabel("Lens Model");
        p->setHint("Parametric: the aberrations are set by hand (Aberrations tab). Real Lens: an optical prescription is "
                   "ray traced through its glasses, wavelength by wavelength (aberrations, distortion, colour, vignetting "
                   "all come from the design).");
        p->appendOption("Parametric");
        p->appendOption("Real Lens");
        p->setDefault(0);
        p->setAnimates(false);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamLens);
        p->setLabel("Lens");
        p->setHint("Built-in prescriptions (public designs, scaled to the focal length), or a .lens file of your own. "
                   "Each lens covers its design field: beyond it (short focal length on a large filmback) the corners vignette.");
        for (int i = 0; i < ca::builtinLensCount(); ++i) {
            ca::LensPrescription pr;
            std::string label = ca::builtinLensId(i);
            if ( ca::parseLensText(ca::builtinLensText(i), pr) ) {
                std::ostringstream o;
                o.precision(3);
                o << pr.name << " (field " << 2. * pr.designHalfFieldDeg << " deg)";
                label = o.str();
            }
            p->appendOption(label);
        }
        p->appendOption("Custom File");
        p->setDefault(0);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        StringParamDescriptor* p = desc.defineStringParam(kParamLensFile);
        p->setLabel("Lens File");
        p->setHint("A .lens prescription (see the lenses folder of the project for the format).");
        p->setStringType(eStringTypeFilePath);
        p->setFilePathExists(true);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamFocusDistance, "Focus Distance (m)", "Real lens: distance of "
                                                "the subject in focus, from the front of the lens (0 = infinity). The aberrations "
                                                "change with the focus distance.", 0., 0., 1e6, 0., 20.);
        p->setLayoutHint(eLayoutHintDivider);
    }
    defineDouble(desc, page, kParamFNumber, "f-number", "Working aperture f/N. Diffraction grows when the aperture closes "
                 "(Airy disc diameter 2.44 x wavelength x N).", 2.8, 0.5, 128., 1., 22.);
    defineDouble(desc, page, kParamFilmback, "Filmback Width (mm)", "Width of the sensor area covered by the image, in mm "
                 "(Super 35: 24.89, full frame: 36). Sets the size of a pixel, hence the size of the PSFs in pixels. "
                 "For an anamorphic plate, the squeezed width on the sensor.", 24.89, 0.1, 200., 5., 70.);
    defineDouble(desc, page, kParamFocal, "Focal Length (mm)", "Focal length. Real lens: the prescription is scaled to it "
                 "(with the filmback, it sets the field of view). Parametric: used for the natural cos^4 falloff.", 50., 1., 2000., 10., 200.);
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamCenterOnImage, "Centre on Image", "The optical axis is "
                                               "the centre of the input image. Uncheck to place it with Optical Center "
                                               "(for a crop or a reframed plate).", true);
        p->setAnimates(false);
    }
    {
        Double2DParamDescriptor* p = desc.defineDouble2DParam(kParamCenter);
        p->setLabel("Optical Center");
        p->setHint("Position of the optical axis (drag it in the viewer), used when Centre on Image is off. "
                   "Aberrations grow from there.");
        p->setDoubleType(eDoubleTypeXYAbsolute);
        if ( p->supportsDefaultCoordinateSystem() ) {
            p->setDefaultCoordinateSystem(eCoordinatesNormalised);
        }
        p->setDefault(0.5, 0.5);
        p->setUseHostNativeOverlayHandle(true);
        p->setLayoutHint(eLayoutHintDivider);
        p->setAnimates(true);
        page->addChild(*p);
    }
    {
        IntParamDescriptor* p = defineInt(desc, page, kParamBlades, "Diaphragm Blades",
                                          "Number of blades (0 = perfectly round). Straight blades give diffraction "
                                          "stars: 2N spikes for an odd number, N for an even number.", 0, 0, 24);
        p->setDisplayRange(0, 16);
    }
    defineDouble(desc, page, kParamRoundness, "Blade Roundness", "0 = straight blades (polygon), 1 = round.", 0.3, 0., 1., 0., 1.);
    defineDouble(desc, page, kParamRotation, "Blade Rotation", "Rotation of the diaphragm, in degrees.", 0., -360., 360., -90., 90.);
    defineDouble(desc, page, kParamPupilAspect, "Pupil Aspect", "Height / width of the pupil: > 1 gives tall oval bokeh "
                 "(anamorphic look).", 1., 0.1, 10., 0.5, 2.);
    defineDouble(desc, page, kParamVignetting, "Mechanical Vignetting", "The lens barrel clips the pupil off axis: cat-eye "
                 "bokeh and darker corners. How deep it cuts into the pupil in the corner, in pupil radii.", 0., 0., 2., 0., 1.);
    defineDouble(desc, page, kParamVignettingStart, "Vignetting Start", "Field position (0 = centre, 1 = corner) where the "
                 "mechanical vignetting begins.", 0.5, 0., 0.95, 0., 0.95);
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamNatural, "Natural Vignetting", "Amount of the natural cos^4 "
                                                "falloff (0 = none, 1 = physical).", 1., 0., 1., 0., 1.);
        (void)p;
    }

    // ---- Aberrations
    page = desc.definePageParam("Aberrations");
    const char* unit = " In waves (at 550 nm), at the edge of the pupil and in the image corner.";
    defineDouble(desc, page, kParamSpherical, "Spherical (W040)", (std::string("Spherical aberration: soft halo around highlights, "
                 "glow wide open.") + unit).c_str(), 1., -100., 100., -10., 10.);
    defineDouble(desc, page, kParamComa, "Coma (W131)", (std::string("Coma: comet-shaped highlights towards the corners. "
                 "Positive: tail towards the centre; negative: tail outwards.") + unit).c_str(), 2., -100., 100., -20., 20.);
    defineDouble(desc, page, kParamAstig, "Astigmatism (W222)", (std::string("Astigmatism: blur stretched radially or "
                 "tangentially towards the corners.") + unit).c_str(), 1.5, -100., 100., -20., 20.);
    defineDouble(desc, page, kParamCurvature, "Field Curvature (W220)", (std::string("Field curvature: the corners go out of "
                 "focus.") + unit).c_str(), 0., -100., 100., -20., 20.);
    defineDouble(desc, page, kParamDistortion, "Distortion (W311)", (std::string("Distortion: negative = barrel, positive = "
                 "pincushion.") + unit).c_str(), 0., -100., 100., -10., 10.);
    defineDouble(desc, page, kParamDefocus, "Defocus (W020)", (std::string("Defocus of the whole image (shows the shape of the "
                 "diaphragm).") + unit).c_str(), 0., -100., 100., -20., 20.);
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamAxialCA, "Axial Colour", "Axial chromatic aberration "
                                                "(focus changes with wavelength): purple/green fringes around sharp edges. "
                                                "Defocus at 486 nm minus 656 nm, in waves (N-BK7 dispersion).", 1., -50., 50., -5., 5.);
        p->setLayoutHint(eLayoutHintNormal);
    }
    defineDouble(desc, page, kParamLateralCA, "Lateral Colour", "Lateral chromatic aberration (magnification changes with "
                 "wavelength): radial colour fringes towards the corners. Image shift between 486 and 656 nm in the corner, "
                 "in waves of wavefront tilt.", 1., -50., 50., -5., 5.);
    defineDouble(desc, page, kParamSpheroChrom, "Spherochromatism", "Change of the spherical aberration with wavelength "
                 "(coloured halos).", 0., -20., 20., -3., 3.);
    {
        StringParamDescriptor* p = desc.defineStringParam(kParamZernike);
        p->setLabel("Zernike Terms");
        p->setHint("Extra wavefront terms 'j:coefficient' (Noll index j >= 4, RMS waves), separated by commas. "
                   "Examples: 9:0.5 (trefoil), 11:0.3 (secondary spherical).");
        p->setDefault("");
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamAmount, "Aberration Amount", "Multiplies every aberration "
                                                "(parametric coefficients, or the wavefront of the real lens; 0 = perfect lens, "
                                                "only diffraction).", 1., 0., 10., 0., 3.);
        p->setLayoutHint(eLayoutHintDivider);
    }
    defineDouble(desc, page, kParamDistortionAmount, "Distortion Amount", "Real lens: multiplies its distortion (0 = rectilinear).",
                 1., 0., 5., 0., 2.);
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamLateralAmount, "Lateral Colour Amount", "Real lens: multiplies "
                                                "its lateral colour (colour fringes towards the corners).", 1., 0., 10., 0., 3.);
        p->setLayoutHint(eLayoutHintDivider);
    }

    // ---- Spectral
    page = desc.definePageParam("Spectral");
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamColorSpace);
        p->setLabel("Colour Space");
        p->setHint("Linear working space of the images (primaries of the RGB values). Each pixel is converted to the "
                   "smoothest spectrum with exactly its RGB values, and back with the CIE 1931 colour matching functions (or the sensitivities of the chosen Camera).");
        p->appendOption("ACEScg");
        p->appendOption("Linear Rec.709 / sRGB");
        p->appendOption("Linear Rec.2020");
        p->setDefault(0);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamCamera);
        p->setLabel("Camera");
        p->setHint("Spectral sensitivities of the camera that shot the plate. The lens separates the wavelengths; the "
                   "camera decides which colour each one gets (the hue of the fringes and of the rainbow edges of "
                   "highlights). Colours without lens effects do not change. CIE 1931 Observer: the colours a human "
                   "eye would see. The built-in cameras come from the Academy's rawtoaces-data (Apache-2.0); Custom "
                   "File reads a text file with lines \"nm r g b\" or a rawtoaces-data JSON file.");
        p->appendOption("CIE 1931 Observer");
        for (int i = 0; i < ca::builtinCameraCount(); ++i) {
            p->appendOption(ca::builtinCameraName(i));
        }
        p->appendOption("Custom File");
        p->setDefault(0);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        StringParamDescriptor* p = desc.defineStringParam(kParamCameraFile);
        p->setLabel("Camera File");
        p->setHint("Camera sensitivities: text file, one line per wavelength \"nm r g b\" (covering at least 420-680 nm), "
                   "or a rawtoaces-data JSON file.");
        p->setStringType(eStringTypeFilePath);
        p->setFilePathExists(true);
        p->setAnimates(false);
        page->addChild(*p);
    }
    defineInt(desc, page, kParamWlPreview, "Wavelengths (Preview)", "Number of wavelengths between 400 and 700 nm in Preview "
              "quality.", 8, 3, 64);
    defineInt(desc, page, kParamWlFinal, "Wavelengths (Final)", "Number of wavelengths in Final quality.", 24, 3, 64);

    // ---- Render
    page = desc.definePageParam("Render");
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamQuality);
        p->setLabel("Quality");
        p->setHint("Preview: coarse PSF grid and few wavelengths, to set the lens interactively. Final: fine grid and more "
                   "wavelengths (same look, finer). The PSFs are interpolated between grid points: about 4 % of error on a "
                   "strongly aberrated lens with a 7 x 5 grid, under 1 % with 15 x 11. Preview in Viewer, Final in Render uses the "
                   "host's interactive flag (Final when rendering to disk); hosts that do not set it always get Final.");
        p->appendOption("Preview");
        p->appendOption("Final");
        p->appendOption("Preview in Viewer, Final in Render");
        p->setDefault(0);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        Int2DParamDescriptor* p = desc.defineInt2DParam(kParamGridPreview);
        p->setLabel("PSF Grid (Preview)");
        p->setHint("Number of PSF positions across and down the image, in Preview quality.");
        p->setDefault(7, 5);
        p->setRange(1, 1, 64, 64);
        p->setDisplayRange(2, 2, 32, 32);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamGridMode);
        p->setLabel("Final Grid");
        p->setHint("Automatic: starting from the Preview grid, the grid is refined until the PSF interpolated in the middle "
                   "of a cell differs from the PSF computed there by less than Grid Tolerance. Fixed: PSF Grid (Final).");
        p->appendOption("Automatic");
        p->appendOption("Fixed");
        p->setDefault(0);
        p->setAnimates(false);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
        page->addChild(*p);
    }
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamGridTolerance, "Grid Tolerance (%)", "Largest interpolation "
                                                "error of the PSFs in Final quality with the automatic grid (L1 difference, "
                                                "after a 0.7 pixel blur that hides sub-pixel differences).", 2., 0.2, 20., 0.5, 5.);
        p->setAnimates(false);
    }
    {
        Int2DParamDescriptor* p = desc.defineInt2DParam(kParamGridFinal);
        p->setLabel("PSF Grid (Final)");
        p->setHint("Number of PSF positions across and down the image, in Final quality.");
        p->setDefault(15, 11);
        p->setRange(1, 1, 64, 64);
        p->setDisplayRange(2, 2, 32, 32);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        IntParamDescriptor* p = defineInt(desc, page, kParamMaxPsf, "Max PSF Size", "Largest PSF, in pixels at full "
                                          "resolution. Light beyond it is lost from the blur (raise it for strong aberrations "
                                          "or defocus).", 129, 9, 513);
        p->setDisplayRange(17, 257);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamAutoPsf, "Auto Size", "Size the PSFs from the lens (largest "
                                               "geometric spot + diffraction), up to Max PSF Size: small and fast for a sharp "
                                               "lens, large enough for a soft one.", true);
        p->setLayoutHint(eLayoutHintDivider);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamUseGPU, "Use GPU (CUDA)", "Render on the NVIDIA GPU when "
                                               "available (same result as the CPU up to float precision).", true);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    {
        // the GPU model, shown as the label's default (Nuke displays the default of label strings)
        StringParamDescriptor* p = desc.defineStringParam(kParamGPUStatus);
        p->setLabels("", "", "");
        p->setHint("GPU used for rendering (first CUDA device of this machine).");
        p->setStringType(eStringTypeLabel);
        p->setIsPersistent(false);
        p->setEvaluateOnChange(false);
        p->setAnimates(false);
        p->setDefault( gpuDescription() );
        p->setLayoutHint(eLayoutHintDivider);
        page->addChild(*p);
    }

    // ---- Output
    page = desc.definePageParam("Output");
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kParamView);
        p->setLabel("View");
        p->setHint("Image: the plate through the lens. PSF Map: the PSFs (white light) at their place in the image, enlarged, "
                   "to set the lens. Test Chart: the lens applied to a built-in chart (Siemens stars and HDR point lights).");
        p->appendOption("Image");
        p->appendOption("PSF Map");
        p->appendOption("Test Chart");
        p->setDefault(0);
        p->setAnimates(false);
        page->addChild(*p);
    }
    {
        IntParamDescriptor* p = defineInt(desc, page, kParamPsfZoom, "PSF Zoom", "Enlargement of the PSFs in the PSF Map view "
                                          "(0 = automatic: the largest PSF fills most of a grid cell).", 0, 0, 64);
        p->setDisplayRange(0, 16);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamPsfBackground, "Image Behind", "PSF Map view: show the "
                                               "image, dimmed, behind the PSFs (off: black background).", true);
        p->setLayoutHint(eLayoutHintDivider);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamBlur, "Blur", "Aberrations and diffraction (PSF).", true);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamDistort, "Distortion & Lateral Colour",
                                               "Chief ray displacement: distortion and lateral colour.", true);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    {
        BooleanParamDescriptor* p = defineBool(desc, page, kParamIllum, "Vignetting",
                                               "Relative illumination: natural falloff and mechanical vignetting.", true);
        p->setLayoutHint(eLayoutHintDivider);
    }
    {
        DoubleParamDescriptor* p = defineDouble(desc, page, kParamMix, "Mix", "Mix with the source.", 1., 0., 1., 0., 1.);
        p->setLayoutHint(eLayoutHintNoNewLine, 1);
    }
    defineBool(desc, page, kParamMaskInvert, "Invert Mask", "Invert the mask.", false);
}

ImageEffect* ChromaAberrationsPluginFactory::createInstance(OfxImageEffectHandle handle, ContextEnum /*context*/)
{
    return new ChromaAberrationsPlugin(handle);
}

static ChromaAberrationsPluginFactory p1(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor);
mRegisterPluginFactoryInstance(p1)
