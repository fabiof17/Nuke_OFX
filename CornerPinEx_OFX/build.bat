@echo off
rem ===========================================================================
rem  CornerPinEx_OFX - Windows build (Visual Studio + CUDA)
rem
rem  Double-click this file, or run it from a command prompt:
rem     build.bat            build (GPU version if the CUDA Toolkit is installed)
rem     build.bat cpu        CPU-only build (no CUDA)
rem     build.bat clean      delete the build folder, then build
rem
rem  Requirements: Visual Studio 2019/2022 (Desktop development with C++), Git,
rem                CUDA Toolkit 12.x for the GPU version.
rem  Result      : build\CornerPinEx_OFX.ofx.bundle
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "WITH_CUDA=ON"
set "CLEAN=0"
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="cpu" set "WITH_CUDA=OFF"
if /i "%~1"=="clean" set "CLEAN=1"
shift
goto parse_args
:args_done

echo.
echo === CornerPinEx_OFX : build ===
echo.

rem --- 1. Visual Studio ------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto no_vs
set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH goto no_vs
if not exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" goto no_vs
echo [1/5] Visual Studio : %VSPATH%
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>&1
if errorlevel 1 goto no_vs

rem --- 2. Tools (CMake and Ninja ship with Visual Studio) ---------------------
where cmake >nul 2>&1
if errorlevel 1 goto no_cmake
where git >nul 2>&1
if errorlevel 1 goto no_git
set "GENERATOR=Ninja"
where ninja >nul 2>&1
if errorlevel 1 set "GENERATOR="
echo [2/5] CMake, Git : OK

rem --- 3. CUDA ---------------------------------------------------------------
set "CUDA_ARGS="
if "%WITH_CUDA%"=="OFF" goto cuda_off
set "NVCC="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC for /f "tokens=*" %%i in ('where nvcc 2^>nul') do if not defined NVCC set "NVCC=%%i"
if not defined NVCC goto no_cuda
echo [3/5] CUDA : %NVCC%
set CUDA_ARGS="-DCMAKE_CUDA_COMPILER=%NVCC:\=/%"

rem Architectures: Turing, Ampere, Ada + PTX; adds the installed GPU if it is newer (e.g. RTX 50xx)
set "ARCHS=75-real;86-real;89-real;75-virtual"
set "CC="
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do if not defined CC set "CC=%%i"
if not defined CC goto archs_done
echo %CC%| findstr /r /c:"^[0-9][0-9]*\.[0-9][0-9]* *$" >nul
if errorlevel 1 goto archs_done
set "CC=%CC:.=%"
set "CC=%CC: =%"
if "%CC%"=="75" goto archs_done
if "%CC%"=="86" goto archs_done
if "%CC%"=="89" goto archs_done
if %CC% LSS 75 goto archs_done
set "ARCHS=%ARCHS%;%CC%-real"
echo       GPU detected: sm_%CC% (added to the build)
:archs_done
set CUDA_ARGS=%CUDA_ARGS% "-DCMAKE_CUDA_ARCHITECTURES=%ARCHS%"
goto cuda_done

:no_cuda
echo [3/5] CUDA Toolkit not found: building the CPU-only version.
echo       For the GPU version, install the CUDA Toolkit 12.x, then run "build.bat clean".
set "WITH_CUDA=OFF"
goto cuda_done
:cuda_off
echo [3/5] CPU-only version, no CUDA
:cuda_done

rem --- 4. Configure and build ------------------------------------------------
if "%CLEAN%"=="1" if exist build rmdir /s /q build
if exist build\CMakeCache.txt goto configured
echo [4/5] Configuring...
if defined GENERATOR goto cfg_ninja
cmake -S . -B build -A x64 -DCORNERPINEX_OFX_WITH_CUDA=%WITH_CUDA% -DCORNERPINEX_OFX_BUILD_TESTS=ON %CUDA_ARGS%
goto cfg_done
:cfg_ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCORNERPINEX_OFX_WITH_CUDA=%WITH_CUDA% -DCORNERPINEX_OFX_BUILD_TESTS=ON %CUDA_ARGS%
:cfg_done
if errorlevel 1 goto failed
:configured
echo [4/5] Building...
cmake --build build --config Release
if errorlevel 1 goto failed

rem --- 5. Tests --------------------------------------------------------------
echo [5/5] Running tests...
ctest --test-dir build -C Release --output-on-failure
if errorlevel 1 goto tests_failed
goto success

:tests_failed
echo.
echo WARNING: some tests failed (see above). The plugin is built, but check it before using it.
goto install

:success
echo.
echo Build succeeded.

rem --- Installation ----------------------------------------------------------
:install
set "BUNDLE=%~dp0build\CornerPinEx_OFX.ofx.bundle"
set "OFXDIR=%CommonProgramFiles%\OFX\Plugins"
echo.
echo Plugin : %BUNDLE%
echo.
net session >nul 2>&1
if errorlevel 1 goto install_manual
choice /c YN /m "Install into %OFXDIR% "
if errorlevel 2 goto done
if not exist "%OFXDIR%" mkdir "%OFXDIR%"
if exist "%OFXDIR%\CornerPinEx_OFX.ofx.bundle" rmdir /s /q "%OFXDIR%\CornerPinEx_OFX.ofx.bundle"
xcopy /e /i /q /y "%BUNDLE%" "%OFXDIR%\CornerPinEx_OFX.ofx.bundle" >nul
if errorlevel 1 goto install_error
echo Installed into %OFXDIR%\CornerPinEx_OFX.ofx.bundle - restart Nuke.
goto done

:install_manual
echo To install: copy the CornerPinEx_OFX.ofx.bundle folder into
echo    %OFXDIR%
echo (or run build.bat "as administrator" to do it automatically).
goto done

:install_error
echo Copy to %OFXDIR% failed. Copy the folder by hand.
goto done

rem --- Errors ----------------------------------------------------------------
:no_vs
echo [ERROR] Visual Studio 2019/2022 with "Desktop development with C++" not found.
echo         https://visualstudio.microsoft.com/downloads/  (Community is enough)
goto end_error
:no_cmake
echo [ERROR] CMake not found. Install the Visual Studio component "C++ CMake tools for Windows",
echo         or CMake from https://cmake.org/download/
goto end_error
:no_git
echo [ERROR] Git not found (needed to download the OFX headers).
echo         https://git-scm.com/download/win
goto end_error
:failed
echo.
echo [ERROR] The build failed (see the messages above).
echo         Tip: "build.bat clean" starts from scratch.
goto end_error

:end_error
echo.
pause
endlocal
exit /b 1

:done
echo.
pause
endlocal
exit /b 0
