# CornerPinEx_OFX - GPU-accelerated CornerPin OFX plugin (for Foundry Nuke and other OFX hosts)
# GPL v2 or later, derived from Natron's openfx-misc CornerPin.
#
# Windows (Visual Studio 2019/2022 + CUDA Toolkit 12.x):
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#   cmake --install build --config Release      (copies the bundle to the OFX plugin folder)

cmake_minimum_required(VERSION 3.20)

project(CornerPinEx_OFX VERSION 1.0.0 LANGUAGES C CXX)

option(CORNERPINEX_OFX_WITH_CUDA "Build the CUDA render engine" ON)
option(CORNERPINEX_OFX_BUILD_TESTS "Build the engine tests (no OFX host needed)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  # Nuke 14 and older ship an older msvcp140.dll: without this, std::mutex built with VS 2022 17.10+ crashes there.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# ---------------------------------------------------------------------------
# Dependencies: the OFX headers/support library and openfx-supportext, pinned to
# the commits used by openfx-misc. Set OFX_DIR / SUPPORTEXT_DIR to use local copies.
include(FetchContent)
set(OFX_DIR "" CACHE PATH "Local checkout of https://github.com/NatronGitHub/openfx (optional)")
set(SUPPORTEXT_DIR "" CACHE PATH "Local checkout of https://github.com/NatronGitHub/openfx-supportext (optional)")

if(NOT OFX_DIR)
  FetchContent_Declare(openfx
    GIT_REPOSITORY https://github.com/NatronGitHub/openfx.git
    GIT_TAG 2303ff811bee3ffe085287602f684fe5fe5357e0
    SOURCE_SUBDIR do-not-add-subdirectory)
  FetchContent_MakeAvailable(openfx)
  set(OFX_DIR ${openfx_SOURCE_DIR})
endif()
if(NOT SUPPORTEXT_DIR)
  FetchContent_Declare(openfx_supportext
    GIT_REPOSITORY https://github.com/NatronGitHub/openfx-supportext.git
    GIT_TAG 2485505dbe55885ee92ca9f164e7732a85856a13
    SOURCE_SUBDIR do-not-add-subdirectory)
  FetchContent_MakeAvailable(openfx_supportext)
  set(SUPPORTEXT_DIR ${openfx_supportext_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Plugin
set(OFX_SUPPORT_SOURCES
  ${OFX_DIR}/Support/Library/ofxsCore.cpp
  ${OFX_DIR}/Support/Library/ofxsImageEffect.cpp
  ${OFX_DIR}/Support/Library/ofxsInteract.cpp
  ${OFX_DIR}/Support/Library/ofxsLog.cpp
  ${OFX_DIR}/Support/Library/ofxsMultiThread.cpp
  ${OFX_DIR}/Support/Library/ofxsParams.cpp
  ${OFX_DIR}/Support/Library/ofxsProperty.cpp
  ${OFX_DIR}/Support/Library/ofxsPropertyValidation.cpp
)
set(SUPPORTEXT_SOURCES
  ${SUPPORTEXT_DIR}/ofxsThreadSuite.cpp
  ${SUPPORTEXT_DIR}/tinythread.cpp
  ${SUPPORTEXT_DIR}/ofxsOGLTextRenderer.cpp
  ${SUPPORTEXT_DIR}/ofxsOGLFontData.cpp
  ${SUPPORTEXT_DIR}/ofxsShutter.cpp
)
set(PLUGIN_SOURCES
  src/CornerPinEx_OFX.cpp
  src/cpTransform3x3.cpp
  src/cpCpuEngine.cpp
)

if(CORNERPINEX_OFX_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Turing, Ampere, Ada (+ PTX for newer GPUs, JIT-compiled by the driver).
      # Add 120-real for RTX 50xx with CUDA >= 12.8, e.g. -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;120-real;75-virtual"
      set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;75-virtual")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND PLUGIN_SOURCES src/cpCuda.cu)
    set(CORNERPINEX_OFX_HAVE_CUDA ON)
  else()
    message(WARNING "CUDA compiler not found: building the CPU-only version")
  endif()
endif()
if(NOT CORNERPINEX_OFX_HAVE_CUDA)
  list(APPEND PLUGIN_SOURCES src/cpCudaStub.cpp)
endif()

add_library(CornerPinEx_OFX MODULE ${PLUGIN_SOURCES} ${OFX_SUPPORT_SOURCES} ${SUPPORTEXT_SOURCES})

target_include_directories(CornerPinEx_OFX PRIVATE
  src
  ${OFX_DIR}/include
  ${OFX_DIR}/Support/include
  ${OFX_DIR}/Support/Plugins/include
  ${SUPPORTEXT_DIR}
)
target_compile_definitions(CornerPinEx_OFX PRIVATE
  OFX_EXTENSIONS_VEGAS OFX_EXTENSIONS_NUKE OFX_EXTENSIONS_NATRON OFX_EXTENSIONS_TUTTLE
  $<$<CONFIG:Release>:NDEBUG>
)
if(WIN32)
  target_compile_definitions(CornerPinEx_OFX PRIVATE WIN32 _WINDOWS NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

# libGL.so.1 (present on every Linux with an OpenGL driver) rather than the GLVND libOpenGL.so.0
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
target_link_libraries(CornerPinEx_OFX PRIVATE OpenGL::GL)
if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(CornerPinEx_OFX PRIVATE Threads::Threads)
endif()

if(CORNERPINEX_OFX_HAVE_CUDA)
  # static CUDA runtime: no cudart DLL to ship, only the NVIDIA driver is needed.
  # Without a driver the plugin still loads and renders on the CPU.
  target_link_libraries(CornerPinEx_OFX PRIVATE CUDA::cudart_static)
  set_target_properties(CornerPinEx_OFX PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  target_compile_options(CornerPinEx_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>")
endif()

if(UNIX AND NOT APPLE)
  # Linux: libstdc++/libgcc linked statically (the plugin does not depend on the libstdc++ shipped
  # with Nuke) and only the OFX entry points exported (no symbol clash with Nuke or other plugins).
  set_target_properties(CornerPinEx_OFX PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LINK_DEPENDS ${CMAKE_SOURCE_DIR}/src/exports.map)
  target_link_options(CornerPinEx_OFX PRIVATE -static-libstdc++ -static-libgcc
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/src/exports.map" -Wl,--exclude-libs,ALL)
endif()

if(MSVC)
  # /Zc:__cplusplus: without it MSVC reports __cplusplus = 199711L, and the OFX support library
  # then falls back to std::auto_ptr, which no longer exists in C++17.
  # /wd5040: the OFX support library uses dynamic exception specifications (harmless warning).
  target_compile_options(CornerPinEx_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W3;/EHsc;/MP;/Zc:__cplusplus;/wd5040;/utf-8>")
  target_compile_options(CornerPinEx_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus;-allow-unsupported-compiler>")
else()
  target_compile_options(CornerPinEx_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-fvisibility=hidden>")
endif()

# OFX bundle layout: CornerPinEx_OFX.ofx.bundle/Contents/<arch>/CornerPinEx_OFX.ofx
if(WIN32)
  set(OFX_ARCH Win64)
elseif(APPLE)
  set(OFX_ARCH MacOS)
else()
  set(OFX_ARCH Linux-x86-64)
endif()
set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/CornerPinEx_OFX.ofx.bundle)
set_target_properties(CornerPinEx_OFX PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/${OFX_ARCH}
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BUNDLE_DIR}/Contents/${OFX_ARCH}
  LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BUNDLE_DIR}/Contents/${OFX_ARCH}
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${BUNDLE_DIR}/Contents/${OFX_ARCH}
)
# icons (named after the plugin identifiers)
add_custom_command(TARGET CornerPinEx_OFX POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_DIR}/Contents/Resources
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/resources/net.sf.openfx.CornerPinPlugin.png ${BUNDLE_DIR}/Contents/Resources/community.openfx.CornerPinEx_OFX.png
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/resources/net.sf.openfx.CornerPinPlugin.png ${BUNDLE_DIR}/Contents/Resources/community.openfx.CornerPinExMasked_OFX.png
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/resources/net.sf.openfx.CornerPinPlugin.svg ${BUNDLE_DIR}/Contents/Resources/community.openfx.CornerPinEx_OFX.svg
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/resources/net.sf.openfx.CornerPinPlugin.svg ${BUNDLE_DIR}/Contents/Resources/community.openfx.CornerPinExMasked_OFX.svg
)

# install: cmake --install build --config Release [--prefix <folder>]
if(WIN32)
  set(OFX_INSTALL_DIR "C:/Program Files/Common Files/OFX/Plugins" CACHE PATH "OFX plugin folder")
else()
  set(OFX_INSTALL_DIR "/usr/OFX/Plugins" CACHE PATH "OFX plugin folder")
endif()
install(DIRECTORY ${BUNDLE_DIR} DESTINATION ${OFX_INSTALL_DIR})

# ---------------------------------------------------------------------------
# Engine tests (CPU engine, and GPU engine if a CUDA device is present)
if(CORNERPINEX_OFX_BUILD_TESTS)
  set(TEST_SOURCES tests/test_engine.cpp src/cpCpuEngine.cpp)
  if(CORNERPINEX_OFX_HAVE_CUDA)
    list(APPEND TEST_SOURCES src/cpCuda.cu)
  else()
    list(APPEND TEST_SOURCES src/cpCudaStub.cpp)
  endif()
  add_executable(test_engine ${TEST_SOURCES})
  if(MSVC)
    target_compile_options(test_engine PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus;/utf-8>"
                                               "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus;-allow-unsupported-compiler>")
  endif()
  target_include_directories(test_engine PRIVATE src)
  if(CORNERPINEX_OFX_HAVE_CUDA)
    target_link_libraries(test_engine PRIVATE CUDA::cudart_static)
    set_target_properties(test_engine PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  enable_testing()
  add_test(NAME test_engine COMMAND test_engine)
endif()
