param(
    [string]$Ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$raw = Join-Path $repo 'build\profiles\stage11'
New-Item -ItemType Directory -Force -Path $raw | Out-Null
Set-Location -LiteralPath $repo

function Capture(
    [string]$Name,
    [string]$Kernel,
    [string]$Executable,
    [string[]]$Arguments
) {
    $profileArgs = @(
        '--set', 'full', '--target-processes', 'all',
        '--kernel-name-base', 'function',
        '--kernel-name', "regex:.*$Kernel.*",
        '--launch-count', '1',
        '--export', (Join-Path $raw $Name),
        '--force-overwrite'
    )
    Write-Output "Capturing $Name"
    & $Ncu @profileArgs $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Compute failed for $Name with exit code $LASTEXITCODE"
    }
}

$reduction = Join-Path $repo 'build\warpforge-benchmark-reduction.exe'
$reductionArgs = @(
    '--size', '16777216', '--block-size', '256', '--operation', 'sum',
    '--profile-only', '--output-dir', 'build\profiles\stage11\reduction-output'
)
Capture 'reduction-interleaved' 'interleaved_kernel' $reduction ($reductionArgs + @('--variant', 'naive_interleaved'))
Capture 'reduction-warp-shuffle' 'warp_shuffle_kernel' $reduction ($reductionArgs + @('--variant', 'warp_shuffle'))

$gemm = Join-Path $repo 'build\warpforge-benchmark-gemm.exe'
$gemmArgs = @(
    '--size', '1024', '--warmups', '0', '--iterations', '1', '--profile-only',
    '--output-dir', 'build\profiles\stage11\gemm-output'
)
Capture 'gemm-naive' 'naive_fp32_kernel' $gemm ($gemmArgs + @('--implementation', 'custom_naive_fp32'))
Capture 'gemm-register-blocked' 'register_blocked_fp32_kernel' $gemm ($gemmArgs + @('--implementation', 'custom_register_blocked_fp32'))

$transformer = Join-Path $repo 'build\warpforge-benchmark-transformer.exe'
$transformerArgs = @(
    '--rows', '4096', '--columns', '1024', '--elements', '1024',
    '--sequence', '8', '--heads', '2', '--head-dim', '16', '--mask-length', '8',
    '--warmups', '0', '--iterations', '1',
    '--output-dir', 'build\profiles\stage11\transformer-output'
)
Capture 'softmax-naive' 'softmax_naive_kernel' $transformer $transformerArgs
Capture 'softmax-warp' 'softmax_warp_kernel' $transformer $transformerArgs
Capture 'rmsnorm-naive' 'rmsnorm_naive_kernel' $transformer $transformerArgs
Capture 'rmsnorm-block' 'rmsnorm_block_kernel' $transformer $transformerArgs
Write-Output 'All eight Nsight Compute captures passed.'
