param(
    [string]$VisualStudioRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools',
    [switch]$RunSanitizer
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$vcvars = Join-Path $VisualStudioRoot 'VC\Auxiliary\Build\vcvars64.bat'
$cmake = Join-Path $VisualStudioRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctest = Join-Path $VisualStudioRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
foreach ($tool in @($vcvars, $cmake, $ctest)) {
    if (-not (Test-Path -LiteralPath $tool)) {
        throw "Missing Windows build tool: $tool"
    }
}
Set-Location -LiteralPath $repo

$buildCommand = 'call "{0}" -vcvars_ver=14.44 && "{1}" --preset windows-msvc-release && "{1}" --build --preset windows-msvc-release && "{2}" --preset windows-msvc-release' -f $vcvars, $cmake, $ctest
& cmd.exe /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) {
    throw "Windows CUDA build or CTest failed with exit code $LASTEXITCODE"
}

if ($RunSanitizer) {
    & (Join-Path $PSScriptRoot 'run_compute_sanitizer.ps1')
}
Write-Output 'Windows CUDA build and tests passed.'
