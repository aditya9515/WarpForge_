param(
    [string]$Nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$raw = Join-Path $repo 'build\profiles\stage11'
New-Item -ItemType Directory -Force -Path $raw | Out-Null
Set-Location -LiteralPath $repo

foreach ($backend in @('custom', 'cublas')) {
    $outputPrefix = Join-Path $raw "miniinfer-$backend"
    & $Nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none `
        --wait=primary --force-overwrite=true `
        --output=$outputPrefix `
        (Join-Path $repo 'build\warpforge-miniinfer.exe') `
        --fixture-dir benchmarks\fixtures\miniinfer_full `
        --backend $backend --warmups 3 --iterations 5 `
        --output-dir build\profiles\stage11\miniinfer-output
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Systems failed for $backend with exit code $LASTEXITCODE"
    }
    & $Nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum `
        --report cuda_gpu_mem_time_sum --report cuda_api_sum `
        --report cuda_api_trace --report cuda_kern_exec_trace `
        --format csv --force-export=true --force-overwrite=true `
        --output (Join-Path $raw "miniinfer-$backend-stats") `
        (Join-Path $raw "miniinfer-$backend.nsys-rep")
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Systems export failed for $backend with exit code $LASTEXITCODE"
    }
}
Write-Output 'Both MiniInfer timelines and CSV exports passed.'
