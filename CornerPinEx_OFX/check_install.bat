@echo off
rem ===========================================================================
rem  CornerPinEx_OFX - installation check for Nuke
rem  Close Nuke, then double-click this file.
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "OFXDIR=%CommonProgramFiles%\OFX\Plugins"
set "BUNDLE=%OFXDIR%\CornerPinEx_OFX.ofx.bundle"
set "DLL=%BUNDLE%\Contents\Win64\CornerPinEx_OFX.ofx"
set "PROBLEMS=0"

echo.
echo === CornerPinEx_OFX : installation check ===
echo.

rem --- 1. build ---------------------------------------------------------------
echo [1] Build
if exist "build\CornerPinEx_OFX.ofx.bundle\Contents\Win64\CornerPinEx_OFX.ofx" goto built_ok
echo     NO: build\CornerPinEx_OFX.ofx.bundle\Contents\Win64\CornerPinEx_OFX.ofx does not exist.
echo     Run build.bat first and check that it prints "Build succeeded".
set "PROBLEMS=1"
goto check_install
:built_ok
echo     OK: the plugin has been built.

rem --- 2. installation --------------------------------------------------------
:check_install
echo.
echo [2] Installation in %OFXDIR%
if exist "%DLL%" goto installed_ok
echo     NO: %DLL% not found.
if exist "%OFXDIR%\CornerPinEx_OFX.ofx" echo     The .ofx file was copied on its own: copy the whole CornerPinEx_OFX.ofx.bundle FOLDER.
if exist "%OFXDIR%\build" echo     The build folder was copied: copy only build\CornerPinEx_OFX.ofx.bundle.
if exist "%OFXDIR%\CornerPinEx_OFX.ofx.bundle\CornerPinEx_OFX.ofx.bundle" echo     The bundle is nested twice (CornerPinEx_OFX.ofx.bundle\CornerPinEx_OFX.ofx.bundle): move it up one level.
if not exist "build\CornerPinEx_OFX.ofx.bundle" goto inst_fail
net session >nul 2>&1
if errorlevel 1 goto inst_need_admin
choice /c YN /m "    Copy build\CornerPinEx_OFX.ofx.bundle into %OFXDIR% now "
if errorlevel 2 goto inst_fail
if not exist "%OFXDIR%" mkdir "%OFXDIR%"
xcopy /e /i /q /y "build\CornerPinEx_OFX.ofx.bundle" "%BUNDLE%" >nul
if exist "%DLL%" goto installed_ok
:inst_need_admin
echo     To copy it automatically, run this file again with "Run as administrator".
:inst_fail
set "PROBLEMS=1"
goto check_env
:installed_ok
echo     OK : %DLL%

rem --- 3. OFX_PLUGIN_PATH -----------------------------------------------------
:check_env
echo.
echo [3] OFX_PLUGIN_PATH variable
if not defined OFX_PLUGIN_PATH goto env_none
echo     OFX_PLUGIN_PATH = %OFX_PLUGIN_PATH%
echo     Nuke also looks for plugins in these folders. It must not start with ";" or be empty.
goto check_load
:env_none
echo     not set (normal: Nuke uses %OFXDIR%)

rem --- 4. load the DLL --------------------------------------------------------
:check_load
echo.
echo [4] Loading the plugin (as Nuke does)
set "TESTDLL=%DLL%"
if not exist "%TESTDLL%" set "TESTDLL=%~dp0build\CornerPinEx_OFX.ofx.bundle\Contents\Win64\CornerPinEx_OFX.ofx"
if not exist "%TESTDLL%" goto skip_load
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools_check_load.ps1" -Path "%TESTDLL%"
if errorlevel 1 set "PROBLEMS=1"
goto check_cache
:skip_load
echo     skipped (plugin not found)

rem --- 5. Nuke OFX cache -----------------------------------------------------
:check_cache
echo.
echo [5] Nuke OFX plugin cache
set "CACHE=%TEMP%\nuke\ofxplugincache"
if not exist "%CACHE%" goto cache_none
tasklist /fi "imagename eq Nuke*" 2>nul | find /i "nuke" >nul
if not errorlevel 1 goto cache_nuke_running
rmdir /s /q "%CACHE%"
echo     Cache deleted: %CACHE%
echo     Nuke rebuilds the plugin list at next launch.
goto summary
:cache_nuke_running
echo     Nuke is running: close it, then run this check again to clear the cache.
set "PROBLEMS=1"
goto summary
:cache_none
echo     no cache found (%CACHE%)

rem --- summary ----------------------------------------------------------------
:summary
echo.
echo ===========================================================================
if "%PROBLEMS%"=="0" goto all_ok
echo Problems were found: fix them (see above), then run this check again.
goto reminders
:all_ok
echo Everything is in place. Start Nuke, press Tab and type "CornerPinEx_OFX".
:reminders
echo.
echo Reminders:
echo  - Nuke Non-commercial does NOT load third-party OFX plugins (Nuke Indie, Nuke, NukeX and Nuke Studio do).
echo  - If the node still does not show up, start Nuke from a command prompt
echo    and look for messages that contain "OFX" or "CornerPinEx_OFX".
echo ===========================================================================
echo.
pause
endlocal
