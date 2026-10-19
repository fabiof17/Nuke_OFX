@echo off
rem ==========================================================================
rem  Build push_IT_OFX.ofx for Windows (Visual Studio + optional CUDA Toolkit)
rem    build.bat          -> CPU + CUDA (if the CUDA Toolkit is installed)
rem    build.bat cpu      -> CPU only
rem    build.bat clean    -> delete the build folder and rebuild
rem  Everything is also written to build.log next to this script.
rem ==========================================================================
setlocal EnableDelayedExpansion

rem Always work in the folder of this script (double-click, shortcut, etc.)
cd /d "%~dp0"

set "LOG=%~dp0build.log"
echo push_IT_OFX build - %DATE% %TIME% > "%LOG%"

echo.
echo === push_IT_OFX : build ===
echo Folder  : %CD%
echo Log     : %LOG%
echo.

rem ---- the script must not run from inside a zip -----------------------------
if not exist "%~dp0CMakeLists.txt" (
    echo [ERROR] CMakeLists.txt not found next to build.bat.
    echo         Extract the zip first ^(right click ^> Extract All^),
    echo         then run build.bat from the extracted folder.
    goto :fail
)
echo "%~dp0" | findstr /I /C:"\Temp\" >nul && (
    echo [WARNING] The script seems to run from a temporary folder.
    echo           If you opened it directly inside the zip, extract the zip first.
    echo.
)

rem ---- options -----------------------------------------------------------------
set "CUDA_OPT=-DPUSHIT_WITH_CUDA=ON"
if /I "%~1"=="cpu" set "CUDA_OPT=-DPUSHIT_WITH_CUDA=OFF"
if /I "%~1"=="clean" (
    echo Deleting the build folder...
    if exist build rmdir /s /q build
)

rem ---- prerequisites -----------------------------------------------------------
where cmake >nul 2>&1
if errorlevel 1 (
    rem CMake shipped with Visual Studio is not in PATH outside the VS prompt: try it
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin 2^>nul`) do set "PATH=%%i;!PATH!"
)
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found.
    echo         Install it from https://cmake.org/download/
    echo         and tick "Add CMake to the system PATH" during the installation.
    goto :fail
)
for /f "tokens=3" %%v in ('cmake --version ^| findstr /C:"cmake version"') do echo CMake  : %%v

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found ^(needed to download the OpenFX SDK^).
    echo         Install it from https://git-scm.com/download/win
    goto :fail
)
echo Git    : OK

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] Visual Studio not found.
    echo         Install Visual Studio Community with the workload
    echo         "Desktop development with C++".
    goto :fail
)
set "VSVER="
for /f "usebackq delims=" %%v in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion`) do set "VSVER=%%v"
if "%VSVER%"=="" (
    echo [ERROR] Visual Studio is installed without the C++ tools.
    echo         Open "Visual Studio Installer" ^> Modify ^> tick
    echo         "Desktop development with C++".
    goto :fail
)
echo Visual Studio : %VSVER%

if defined CUDA_PATH (
    echo CUDA   : %CUDA_PATH%
) else (
    echo CUDA   : not found ^(no CUDA_PATH^) - CPU-only build
    set "CUDA_OPT=-DPUSHIT_WITH_CUDA=OFF"
)
echo.

rem ---- configure + build -------------------------------------------------------
rem No -G: CMake picks the newest Visual Studio installed (2019, 2022, 2026...).
echo [1/2] CMake configuration...
cmake -S . -B build -A x64 %CUDA_OPT% >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] CMake configuration failed. Log excerpt:
    echo ------------------------------------------------------------
    powershell -NoProfile -Command "Get-Content -Tail 30 '%LOG%'"
    echo ------------------------------------------------------------
    echo If you changed your Visual Studio version, run: build.bat clean
    goto :fail
)

echo [2/2] Compiling ^(may take a minute^)...
cmake --build build --config Release --parallel >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] Compilation failed. Log excerpt:
    echo ------------------------------------------------------------
    powershell -NoProfile -Command "Get-Content '%LOG%' | Select-String -Pattern 'error' | Select-Object -Last 25"
    echo ------------------------------------------------------------
    goto :fail
)

if not exist "build\push_IT_OFX.ofx.bundle\Contents\Win64\push_IT_OFX.ofx" (
    echo [ERROR] Compilation finished but push_IT_OFX.ofx was not found.
    goto :fail
)

echo.
echo === OK ===
echo Plug-in : %CD%\build\push_IT_OFX.ofx.bundle
echo Copy this folder into "C:\Program Files\Common Files\OFX\Plugins"
echo.
pause
exit /b 0

:fail
echo.
echo Full details in: %LOG%
echo.
pause
exit /b 1
