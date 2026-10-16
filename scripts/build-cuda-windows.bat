@echo off
rem Builds cosmic-cli and cosmic-server (fork of trellis.cpp) with CUDA on Windows
rem (MSVC 2022 BuildTools + CUDA 13.x + Ninja, both bundled with VS BuildTools).
rem Usage: scripts\build-cuda-windows.bat [CUDA archs, default 120 = RTX 50xx]
rem        scripts\build-cuda-windows.bat "86;89;120"   to cover more cards
rem Override the VS location with COSMIC_VS=<BuildTools folder>.
setlocal
set "ARCHS=%~1"
if "%ARCHS%"=="" set "ARCHS=120"

if "%COSMIC_VS%"=="" set "COSMIC_VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%COSMIC_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%COSMIC_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%COSMIC_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

cd /d "%~dp0.."
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_OPENMP=OFF ^
  "-DCMAKE_CUDA_ARCHITECTURES=%ARCHS%" || exit /b 1
cmake --build build-cuda --target trellis-cli trellis-server --parallel || exit /b 1
echo.
echo Done: build-cuda\cosmic-cli.exe and build-cuda\cosmic-server.exe
