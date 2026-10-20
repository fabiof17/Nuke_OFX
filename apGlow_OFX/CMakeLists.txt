cmake_minimum_required(VERSION 3.20)
project(apGlow_OFX VERSION 1.0 LANGUAGES CXX)

# ------------------------------------------------------------------ options
option(APGLOW_WITH_CUDA "Build the CUDA GPU back-end" ON)
option(APGLOW_AVX2 "Use AVX2/FMA instructions on the CPU (about 15 % faster, the plug-in will not load on CPUs without AVX2)" OFF)
set(APGLOW_CUDA_ARCHS "" CACHE STRING
    "GPU architectures, e.g. 75-real;86-real;89-real;120-real (empty = automatic from the CUDA version)")
set(OPENFX_ROOT "" CACHE PATH "Local OpenFX SDK checkout (empty = download OFX_Release_1.5.1 from GitHub)")
set(OPENFX_GIT_TAG "OFX_Release_1.5.1" CACHE STRING "OpenFX tag to download when OPENFX_ROOT is empty")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Nuke 14 and older ship their own, older msvcp140.dll, which the plug-in then uses. Code built
# with Visual Studio 2022 17.10 or newer crashes there as soon as a std::mutex is locked
# (constexpr mutex constructor, Microsoft STL change). This keeps the old layout: the plug-ins
# then run with any MSVC runtime, old or new.
if(MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------ OpenFX SDK
if(NOT OPENFX_ROOT)
    include(FetchContent)
    FetchContent_Declare(openfx
        GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
        GIT_TAG        ${OPENFX_GIT_TAG}
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  _no_cmake_)       # sources only: we build the Support library ourselves
    FetchContent_MakeAvailable(openfx)
    set(OPENFX_ROOT ${openfx_SOURCE_DIR})
endif()
if(NOT EXISTS "${OPENFX_ROOT}/Support/include/ofxsImageEffect.h")
    message(FATAL_ERROR "OpenFX SDK not found in OPENFX_ROOT=${OPENFX_ROOT}")
endif()

file(GLOB OFX_SUPPORT_SOURCES "${OPENFX_ROOT}/Support/Library/*.cpp")
add_library(OfxSupport STATIC ${OFX_SUPPORT_SOURCES})
target_include_directories(OfxSupport PUBLIC
    "${OPENFX_ROOT}/include"
    "${OPENFX_ROOT}/Support/include"
    "${OPENFX_ROOT}/Support/Library")
if(MSVC)
    target_compile_definitions(OfxSupport PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# ------------------------------------------------------------------ plug-in
add_library(apGlow_OFX MODULE
    src/GlowPlugin.cpp
    src/GlowCPU.cpp
    src/GlowCommon.h
    src/GlowCore.h)
target_link_libraries(apGlow_OFX PRIVATE OfxSupport)
# viewer overlay (OFX overlay interact) drawn with OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(apGlow_OFX PRIVATE OpenGL::GL)

set(APGLOW_HAS_CUDA OFF)
if(APGLOW_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(APGLOW_HAS_CUDA ON)
        # Architectures are set on the target itself: depending on the generator
        # (Visual Studio) and CUDA version, CMAKE_CUDA_ARCHITECTURES can end up empty.
        if(APGLOW_CUDA_ARCHS)
            set(PS_CUDA_ARCHS "${APGLOW_CUDA_ARCHS}")
        elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            # Turing, Ampere, Ada, Blackwell (native) + PTX for future GPUs
            set(PS_CUDA_ARCHS "75-real;86-real;89-real;120-real;120-virtual")
        else()
            set(PS_CUDA_ARCHS "75-real;86-real;89-real;89-virtual")
        endif()
        message(STATUS "apGlow_OFX: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures = ${PS_CUDA_ARCHS}")
    else()
        message(WARNING "CUDA toolkit not found: building the CPU-only version.")
    endif()
endif()

if(APGLOW_HAS_CUDA)
    target_sources(apGlow_OFX PRIVATE src/GlowCUDA.cu)
    set_target_properties(apGlow_OFX PROPERTIES
        CUDA_ARCHITECTURES "${PS_CUDA_ARCHS}"
        CUDA_RUNTIME_LIBRARY Static          # no cudart DLL to ship next to the plug-in
        CUDA_STANDARD 17)
    target_compile_options(apGlow_OFX PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
else()
    target_sources(apGlow_OFX PRIVATE src/GlowCUDAStub.cpp)
endif()

if(APGLOW_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|amd64")
    if(MSVC)
        target_compile_options(apGlow_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    else()
        target_compile_options(apGlow_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
    endif()
endif()

if(MSVC)
    target_compile_options(apGlow_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /fp:fast>)
    target_compile_definitions(apGlow_OFX PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(apGlow_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3>)
    find_package(Threads REQUIRED)
    target_link_libraries(apGlow_OFX PRIVATE Threads::Threads)
    # Export only the OFX entry points and embed the C++ runtime, so the plug-in
    # does not clash with the libstdc++ shipped inside Nuke.
    set_target_properties(apGlow_OFX OfxSupport PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if(NOT APPLE)
        target_link_options(apGlow_OFX PRIVATE -static-libstdc++ -static-libgcc
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
        set_target_properties(apGlow_OFX PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
    endif()
endif()

# OFX bundle layout: apGlow_OFX.ofx.bundle/Contents/<arch>/apGlow_OFX.ofx
if(WIN32)
    set(OFX_ARCH Win64)
elseif(APPLE)
    set(OFX_ARCH MacOS)
else()
    set(OFX_ARCH Linux-x86-64)
endif()
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/apGlow_OFX.ofx.bundle")
set_target_properties(apGlow_OFX PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    LIBRARY_OUTPUT_DIRECTORY "${BUNDLE_DIR}/Contents/${OFX_ARCH}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BUNDLE_DIR}/Contents/${OFX_ARCH}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${BUNDLE_DIR}/Contents/${OFX_ARCH}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BUNDLE_DIR}/Contents/${OFX_ARCH}")

# Install straight into the standard OFX folder:  cmake --install build
if(WIN32)
    set(OFX_INSTALL_DIR "C:/Program Files/Common Files/OFX/Plugins" CACHE PATH "OFX plug-in folder")
else()
    set(OFX_INSTALL_DIR "/usr/OFX/Plugins" CACHE PATH "OFX plug-in folder")
endif()
install(TARGETS apGlow_OFX LIBRARY DESTINATION "${OFX_INSTALL_DIR}/apGlow_OFX.ofx.bundle/Contents/${OFX_ARCH}")

message(STATUS "apGlow_OFX: CUDA back-end = ${APGLOW_HAS_CUDA}")
