#!/usr/bin/env bash
# ofx_bench: builds and runs the benchmark (Linux). Results in bench_result.txt.
set -e
cd "$(dirname "$0")"
command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > build.log 2>&1
cmake --build build --parallel "$(nproc)" >> build.log 2>&1
./build/ofx_bench 5 | tee bench_result.txt
