cmake_minimum_required(VERSION 3.20)
project(push_IT_OFX VERSION 1.0 LANGUAGES CXX)

# ------------------------------------------------------------------ options
option(PUSHIT_WITH_CUDA "Build the CUDA GPU back-end" ON)
set(PUSHIT_CUDA_ARCHS "" CACHE STRING
    "GPU architectures, e.g. 75-real;86-real;89-real;120-real (empty = automatic from the CUDA version)")
set(OPENFX_ROOT "" CACHE PATH "Local OpenFX SDK checkout (empty = download OFX_Release_1.5.1 from GitHub)")
set(OPENFX_GIT_TAG "OFX_Release_1.5.1" CACHE STRING "OpenFX tag to download when OPENFX_ROOT is empty")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
add_library(push_IT_OFX MODULE
    src/push_IT_OFXPlugin.cpp
    src/push_IT_OFXCPU.cpp
    src/push_IT_OFXCommon.h
    src/push_IT_OFXCore.h)
target_link_libraries(push_IT_OFX PRIVATE OfxSupport)

set(PUSHIT_HAS_CUDA OFF)
if(PUSHIT_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(PUSHIT_HAS_CUDA ON)
        # Architectures are set on the target itself: depending on the generator
        # (Visual Studio) and CUDA version, CMAKE_CUDA_ARCHITECTURES can end up empty.
        if(PUSHIT_CUDA_ARCHS)
            set(PS_CUDA_ARCHS "${PUSHIT_CUDA_ARCHS}")
        elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            # Turing, Ampere, Ada, Blackwell (native) + PTX for future GPUs
            set(PS_CUDA_ARCHS "75-real;86-real;89-real;120-real;120-virtual")
        else()
            set(PS_CUDA_ARCHS "75-real;86-real;89-real;89-virtual")
        endif()
        message(STATUS "push_IT_OFX: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures = ${PS_CUDA_ARCHS}")
    else()
        message(WARNING "CUDA toolkit not found: building the CPU-only version.")
    endif()
endif()

if(PUSHIT_HAS_CUDA)
    target_sources(push_IT_OFX PRIVATE src/push_IT_OFXCUDA.cu)
    set_target_properties(push_IT_OFX PROPERTIES
        CUDA_ARCHITECTURES "${PS_CUDA_ARCHS}"
        CUDA_RUNTIME_LIBRARY Static          # no cudart DLL to ship next to the plug-in
        CUDA_STANDARD 17)
    # no --use_fast_math and no fused multiply-add: the Vector Warp march must compute
    # exactly the same sample positions as on the CPU
    target_compile_options(push_IT_OFX PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --fmad=false>)
else()
    target_sources(push_IT_OFX PRIVATE src/push_IT_OFXCUDAStub.cpp)
endif()

if(MSVC)
    target_compile_options(push_IT_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /fp:fast>)
    target_compile_definitions(push_IT_OFX PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(push_IT_OFX PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3>)
    find_package(Threads REQUIRED)
    target_link_libraries(push_IT_OFX PRIVATE Threads::Threads)
    # Export only the OFX entry points and embed the C++ runtime, so the plug-in
    # does not clash with the libstdc++ shipped inside Nuke.
    set_target_properties(push_IT_OFX OfxSupport PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    if(NOT APPLE)
        target_link_options(push_IT_OFX PRIVATE -static-libstdc++ -static-libgcc
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
        set_target_properties(push_IT_OFX PROPERTIES LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
    endif()
endif()

# OFX bundle layout: push_IT_OFX.ofx.bundle/Contents/<arch>/push_IT_OFX.ofx
if(WIN32)
    set(OFX_ARCH Win64)
elseif(APPLE)
    set(OFX_ARCH MacOS)
else()
    set(OFX_ARCH Linux-x86-64)
endif()
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/push_IT_OFX.ofx.bundle")
set_target_properties(push_IT_OFX PROPERTIES
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
install(TARGETS push_IT_OFX LIBRARY DESTINATION "${OFX_INSTALL_DIR}/push_IT_OFX.ofx.bundle/Contents/${OFX_ARCH}")

message(STATUS "push_IT_OFX: CUDA back-end = ${PUSHIT_HAS_CUDA}")
