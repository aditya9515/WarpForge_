param(
    [string]$Sanitizer = 'compute-sanitizer'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo

$cases = @(
    @{ Tool = 'memcheck'; Executable = 'build\warpforge_runtime_test.exe'; Arguments = @('--skip-allocation-failure') },
    @{ Tool = 'racecheck'; Executable = 'build\warpforge_runtime_test.exe'; Arguments = @('--skip-allocation-failure') },
    @{ Tool = 'synccheck'; Executable = 'build\warpforge_runtime_test.exe'; Arguments = @('--skip-allocation-failure') },
    @{ Tool = 'memcheck'; Executable = 'build\warpforge_miniinfer_test.exe'; Arguments = @('tests\fixtures\miniinfer_small') }
)

foreach ($case in $cases) {
    if (-not (Test-Path -LiteralPath $case.Executable)) {
        throw "Build the CUDA tests before sanitizing: $($case.Executable)"
    }
    Write-Output "Compute Sanitizer $($case.Tool): $($case.Executable)"
    & $Sanitizer --tool $case.Tool --error-exitcode 99 `
        $case.Executable @($case.Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Compute Sanitizer $($case.Tool) failed with exit code $LASTEXITCODE"
    }
}
Write-Output 'Focused Compute Sanitizer checks passed.'
