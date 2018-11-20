:: (c) 2018 ELVEES NeoTek JSC. All rights reserved. 
:: Скрипт используется для сборки библиотеки darknet(yolo) 

if not defined VS140COMNTOOLS goto VS2015_not_installed
if not defined THIRD_PARTY_BUILD_ROOT goto :Third_party_build_root_var_undefined
set OUT_BRANCHNAME=%~dp0
::remove trailing slash and all-except last path components from OUT_BRANCHNAME
for %%d in (%OUT_BRANCHNAME:~0,-1%) do set OUT_BRANCHNAME=%%~nxd
set BASE_OUTPUT_PATH=%THIRD_PARTY_BUILD_ROOT%\yolo\%OUT_BRANCHNAME%
set BASE_INT_PATH=%THIRD_PARTY_BUILD_ROOT%\yolo\%OUT_BRANCHNAME%\build
set CUDA_PATH=%~dp0\..\..\large\cuda\8.0.61.2\win

call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x86

call :build "Debug" "dll" "x64" || goto :build_failed
call :build "Release" "dll" "x64" || goto :build_failed
call :build "Release" "dll_no_gpu" "x64" || goto :build_failed
call :build "Debug" "dll_no_gpu" "x64" || goto :build_failed
call :build "Release" "dll_no_gpu" "x86" || goto :build_failed
call :build "Debug" "dll_no_gpu" "x86" || goto :build_failed
goto :end

::%1= Configuration name with spaces
::%2= Version gpu/no_gpu
::%3= Platform x64/x86
:build
    set CONFIG=%~1
    set USEGPU=%~2
        set PLATFORM=%~3
    msbuild.exe build\darknet\yolo_cpp_%USEGPU%.sln /t:Rebuild /p:BuildTimeExtraCPPDefinitions="DONT_SUPPORT_GPU_TRAIN;" /p:BuildTimeCudaCapailities="compute_20,compute_20;" /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% /p:OutDir=%BASE_OUTPUT_PATH%\%USEGPU%_%CONFIG%_%PLATFORM%\ /p:IntDir=%BASE_INT_PATH%\%USEGPU%_%CONFIG%_%PLATFORM%\
    if not "0"=="%errorlevel%" exit /b "%errorlevel%"
goto :EOF 

:build_failed
    cd "%~dp0"
    echo Error - build failed.
    pause
    exit /b 1

:Third_party_build_root_var_undefined
    echo Error - THIRD_PARTY_BUILD_ROOT undefined.
    pause
    exit /b 1

:VS2015_not_installed
    echo Error - VS140COMNTOOLS environment variable is not defined.
    echo Visual Studio 2015 must be installed.
    pause
    exit /b 1

:end