# ChromaAberrations_OFX - physically based optical aberrations for Nuke (OpenFX)
# Dual-licensed: GPL v3 or commercial (see LICENSING.md).
#
# Builds the OpenFX plug-in, the psfmap and carender tools and the tests:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#   ctest --test-dir build -C Release --output-on-failure

cmake_minimum_required(VERSION 3.20)
project(ChromaAberrations_OFX VERSION 0.2.0 LANGUAGES C CXX)

option(CA_WITH_CUDA "Build the CUDA renderer (needs the CUDA Toolkit)" ON)
option(CA_BUILD_PLUGIN "Build the OpenFX plugin (downloads the OpenFX headers with Git)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(WIN32)
  # Nuke 14 and older ship an older msvcp140.dll: without this, std::mutex built with VS 2022 17.10+ crashes there.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ---- CUDA
set(CA_HAVE_CUDA OFF)
if(CA_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Turing, Ampere, Ada + PTX for newer cards (build.bat adds the installed card if newer)
      set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;75-virtual")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CA_HAVE_CUDA ON)
  else()
    message(WARNING "CUDA compiler not found: building the CPU-only version")
  endif()
endif()

# ---- core library (CPU)
add_library(ca_core STATIC
  src/core/spectrum.cpp
  src/core/aperture.cpp
  src/core/wavefront.cpp
  src/core/psf.cpp
  src/core/psfbank.cpp
  src/core/exrwriter.cpp
  src/core/spectralbasis.cpp
  src/core/camera.cpp
  src/core/lensplan.cpp
  src/core/render.cpp
  src/core/views.cpp
)
target_include_directories(ca_core PUBLIC src/core third_party/pocketfft)
target_link_libraries(ca_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(ca_core PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/W3;/EHsc;/Zc:__cplusplus;/utf-8>")
  target_compile_definitions(ca_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
else()
  target_compile_options(ca_core PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
endif()

# ---- real lenses: glass catalogue, prescriptions, ray tracing
add_library(ca_lens STATIC
  src/lens/glass.cpp
  src/lens/lensfile.cpp
  src/lens/reallens.cpp
)
target_include_directories(ca_lens PUBLIC src/lens)
target_link_libraries(ca_lens PUBLIC ca_core)

# ---- GPU renderer (or its stub)
if(CA_HAVE_CUDA)
  add_library(ca_gpu STATIC src/cuda/render_gpu.cu)
  target_link_libraries(ca_gpu PUBLIC ca_core CUDA::cudart_static)
  set_target_properties(ca_gpu PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  target_compile_options(ca_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
  if(MSVC)
    target_compile_options(ca_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus;-allow-unsupported-compiler>")
  endif()
  if(WIN32)
    # cuFFT has no static library on Windows: cufft64_*.dll is copied next to the programs and
    # into the plugin bundle. The plugin loads it from its own folder before first use.
    target_link_libraries(ca_gpu PUBLIC CUDA::cufft)
    file(GLOB CA_CUFFT_DLLS "${CUDAToolkit_BIN_DIR}/cufft64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cufft64_*.dll")
    list(GET CA_CUFFT_DLLS 0 CA_CUFFT_DLL)
    get_filename_component(CA_CUFFT_DLL_NAME "${CA_CUFFT_DLL}" NAME)
    message(STATUS "cuFFT DLL: ${CA_CUFFT_DLL}")
    target_compile_definitions(ca_gpu PRIVATE CA_CUFFT_DLL="${CA_CUFFT_DLL_NAME}")
  else()
    target_link_libraries(ca_gpu PUBLIC CUDA::cufft_static_nocallback CUDA::culibos)
  endif()
else()
  add_library(ca_gpu STATIC src/core/render_gpu_stub.cpp)
  target_link_libraries(ca_gpu PUBLIC ca_core)
endif()

function(ca_copy_cufft target)
  if(CA_HAVE_CUDA AND WIN32 AND CA_CUFFT_DLL)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CA_CUFFT_DLL}" "$<TARGET_FILE_DIR:${target}>")
  endif()
endfunction()

# ---- tools
add_executable(psfmap tools/psfmap.cpp)
target_include_directories(psfmap PRIVATE tools)
target_link_libraries(psfmap PRIVATE ca_lens)

add_library(ca_tinyexr STATIC third_party/tinyexr/miniz.c)
target_include_directories(ca_tinyexr PUBLIC third_party/tinyexr)
if(MSVC)
  target_compile_options(ca_tinyexr PRIVATE /W0)
else()
  target_compile_options(ca_tinyexr PRIVATE -w)
endif()

add_executable(carender tools/carender.cpp)
target_include_directories(carender PRIVATE tools)
target_link_libraries(carender PRIVATE ca_gpu ca_lens ca_tinyexr)
if(NOT MSVC)
  set_source_files_properties(tools/carender.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-function;-Wno-unused-variable")
endif()
ca_copy_cufft(carender)

# ---- tests
add_executable(test_psf tests/test_psf.cpp)
target_link_libraries(test_psf PRIVATE ca_core)

add_executable(test_render tests/test_render.cpp)
target_link_libraries(test_render PRIVATE ca_gpu)
ca_copy_cufft(test_render)

add_executable(test_lens tests/test_lens.cpp)
target_link_libraries(test_lens PRIVATE ca_lens)

enable_testing()
add_test(NAME test_psf COMMAND test_psf)
add_test(NAME test_lens COMMAND test_lens ${CMAKE_SOURCE_DIR}/tests/reference)
add_test(NAME test_render COMMAND test_render)

# ---- OpenFX plugin
if(CA_BUILD_PLUGIN)
  # OpenFX headers and C++ support library (BSD 3-Clause), pinned. Set OFX_DIR to use a local copy.
  include(FetchContent)
  set(OFX_DIR "" CACHE PATH "Local checkout of https://github.com/NatronGitHub/openfx (optional)")
  if(NOT OFX_DIR)
    FetchContent_Declare(openfx
      GIT_REPOSITORY https://github.com/NatronGitHub/openfx.git
      GIT_TAG 2303ff811bee3ffe085287602f684fe5fe5357e0
      SOURCE_SUBDIR do-not-add-subdirectory)
    FetchContent_MakeAvailable(openfx)
    set(OFX_DIR ${openfx_SOURCE_DIR})
  endif()
  add_library(ChromaAberrations_OFX MODULE
    src/ofx/ChromaAberrations_OFX.cpp
    ${OFX_DIR}/Support/Library/ofxsCore.cpp
    ${OFX_DIR}/Support/Library/ofxsImageEffect.cpp
    ${OFX_DIR}/Support/Library/ofxsInteract.cpp
    ${OFX_DIR}/Support/Library/ofxsLog.cpp
    ${OFX_DIR}/Support/Library/ofxsMultiThread.cpp
    ${OFX_DIR}/Support/Library/ofxsParams.cpp
    ${OFX_DIR}/Support/Library/ofxsProperty.cpp
    ${OFX_DIR}/Support/Library/ofxsPropertyValidation.cpp
  )
  target_include_directories(ChromaAberrations_OFX PRIVATE ${OFX_DIR}/include ${OFX_DIR}/Support/include ${OFX_DIR}/Support/Plugins/include)
  target_compile_definitions(ChromaAberrations_OFX PRIVATE OFX_EXTENSIONS_NUKE OFX_EXTENSIONS_NATRON $<$<CONFIG:Release>:NDEBUG>)
  target_link_libraries(ChromaAberrations_OFX PRIVATE ca_gpu ca_lens)
  if(WIN32)
    target_compile_definitions(ChromaAberrations_OFX PRIVATE WIN32 _WINDOWS _CRT_SECURE_NO_WARNINGS)
  endif()
  if(MSVC)
    # /Zc:__cplusplus: else MSVC reports C++98 and the OFX support library falls back to std::auto_ptr
    # /wd5040: dynamic exception specifications in the OFX support library
    target_compile_options(ChromaAberrations_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W3;/EHsc;/MP;/Zc:__cplusplus;/wd5040;/utf-8>")
    if(CA_HAVE_CUDA AND CA_CUFFT_DLL_NAME)
      # cuFFT is loaded on first use, from the bundle: without it (or without a GPU) the plugin still loads (CPU)
      target_link_options(ChromaAberrations_OFX PRIVATE "/DELAYLOAD:${CA_CUFFT_DLL_NAME}")
      target_link_libraries(ChromaAberrations_OFX PRIVATE delayimp)
    endif()
  else()
    target_compile_options(ChromaAberrations_OFX PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>")
  endif()
  if(UNIX AND NOT APPLE)
    # libstdc++/libgcc linked statically, only the OFX entry points exported (no clash with Nuke's libraries)
    set_target_properties(ChromaAberrations_OFX PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
      LINK_DEPENDS ${CMAKE_SOURCE_DIR}/src/ofx/exports.map)
    target_link_options(ChromaAberrations_OFX PRIVATE -static-libstdc++ -static-libgcc
      "-Wl,--version-script=${CMAKE_SOURCE_DIR}/src/ofx/exports.map" -Wl,--exclude-libs,ALL)
  endif()
  if(WIN32)
    set(OFX_ARCH Win64)
  elseif(APPLE)
    set(OFX_ARCH MacOS)
  else()
    set(OFX_ARCH Linux-x86-64)
  endif()
  set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/ChromaAberrations_OFX.ofx.bundle)
  set_target_properties(ChromaAberrations_OFX PROPERTIES PREFIX "" SUFFIX ".ofx"
    LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/${OFX_ARCH}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BUNDLE_DIR}/Contents/${OFX_ARCH}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BUNDLE_DIR}/Contents/${OFX_ARCH}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${BUNDLE_DIR}/Contents/${OFX_ARCH})
  ca_copy_cufft(ChromaAberrations_OFX)
  # example presets, read by the plugin from Contents/Resources/presets
  add_custom_command(TARGET ChromaAberrations_OFX POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/presets" "${BUNDLE_DIR}/Contents/Resources/presets")
  if(NOT WIN32)
    target_link_libraries(ChromaAberrations_OFX PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(WIN32)
    set(OFX_INSTALL_DIR "C:/Program Files/Common Files/OFX/Plugins" CACHE PATH "OFX plugin folder")
  else()
    set(OFX_INSTALL_DIR "/usr/OFX/Plugins" CACHE PATH "OFX plugin folder")
  endif()
  install(DIRECTORY ${BUNDLE_DIR} DESTINATION ${OFX_INSTALL_DIR})
endif()
