@echo off
rem ==========================================================================
rem  ofx_bench: builds the benchmark, then runs it.
rem  Results are shown and saved to bench_result.txt next to this script.
rem  Needs: Visual Studio (C++), CMake, CUDA Toolkit (same as the plug-ins).
rem ==========================================================================
setlocal
cd /d "%~dp0"

where cmake >nul 2>&1
if errorlevel 1 (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin 2^>nul`) do set "PATH=%%i;%PATH%"
)
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found.
    goto :fail
)

echo [1/2] Building the benchmark...
cmake -S . -B build -A x64 > build.log 2>&1 || goto :buildfail
cmake --build build --config Release --parallel >> build.log 2>&1 || goto :buildfail

echo [2/2] Running ^(about one minute, the GPU is used at full load^)...
echo.
build\Release\ofx_bench.exe 5 > bench_result.txt 2>&1
type bench_result.txt
echo.
echo Results saved to: %~dp0bench_result.txt
echo.
pause
exit /b 0

:buildfail
echo [ERROR] Build failed. Last lines of build.log:
powershell -NoProfile -Command "Get-Content -Tail 25 'build.log'"
:fail
echo.
pause
exit /b 1
