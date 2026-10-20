cmake_minimum_required(VERSION 3.20)
project(ofx_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Nuke 14 and older ship their own, older msvcp140.dll, which the plug-in then uses. Code built
# with Visual Studio 2022 17.10 or newer crashes there as soon as a std::mutex is locked
# (constexpr mutex constructor, Microsoft STL change). This keeps the old layout: the plug-ins
# then run with any MSVC runtime, old or new.
if(MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(ofx_bench
    src/bench.cpp
    src/push_it/push_IT_OFXCPU.cpp
    src/expo_glow/GlowCPU.cpp
    src/l_despill/DespillCPU.cpp
    src/despill_it/DespillITCPU.cpp
    src/master_keyer/MasterKeyerCPU.cpp)
target_include_directories(ofx_bench PRIVATE src)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        set(ARCHS "75-real;86-real;89-real;120-real;120-virtual")
    else()
        set(ARCHS "75-real;86-real;89-real;89-virtual")
    endif()
    target_sources(ofx_bench PRIVATE src/push_it/push_IT_OFXCUDA.cu src/expo_glow/GlowCUDA.cu src/l_despill/DespillCUDA.cu src/despill_it/DespillITCUDA.cu src/master_keyer/MasterKeyerCUDA.cu src/transfer.cu)
    set_target_properties(ofx_bench PROPERTIES
        CUDA_ARCHITECTURES "${ARCHS}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_STANDARD 17)
    # same flags as the plug-ins: push_IT, L_Despill, despill_IT and MasterKeyer without fast-math / FMA, apGlow with fast-math
    set_source_files_properties(src/push_it/push_IT_OFXCUDA.cu src/l_despill/DespillCUDA.cu src/despill_it/DespillITCUDA.cu src/master_keyer/MasterKeyerCUDA.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
    set_source_files_properties(src/expo_glow/GlowCUDA.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")
    message(STATUS "ofx_bench: CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    message(WARNING "CUDA toolkit not found: CPU-only benchmark.")
    target_sources(ofx_bench PRIVATE src/push_it/push_IT_OFXCUDAStub.cpp src/expo_glow/GlowCUDAStub.cpp src/l_despill/DespillCUDAStub.cpp src/despill_it/DespillITCUDAStub.cpp src/master_keyer/MasterKeyerCUDAStub.cpp src/transfer_stub.cpp)
endif()

if(MSVC)
    target_compile_options(ofx_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:fast>)
    target_compile_definitions(ofx_bench PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ofx_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
    find_package(Threads REQUIRED)
    target_link_libraries(ofx_bench PRIVATE Threads::Threads)
endif()
