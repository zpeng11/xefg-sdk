#requires -Version 5.1
<#
.SYNOPSIS
Builds the local XefgProbe target, captures one XeFG run, checks the tensor
trace, and disassembles the captured DXIL.

.DESCRIPTION
OutDir defaults to a new unique directory under the system temporary directory.
BuildDir defaults to $env:TEMP\xefg-sdk-capture-build and may be reused.
Use -Mode 1 for UI off (XEFG_SWAPCHAIN_UI_MODE_NONE) or -Mode 3 for hudless
colour plus UI (XEFG_SWAPCHAIN_UI_MODE_HUDLESS_UITEXTURE).

The capture always enables trace and tensor snapshots. It deliberately skips the
provider model-catalog dump because export only needs trace rows, shader/parameter
relay files, tensors, and DXC disassembly.
#>
[CmdletBinding()]
param(
    [string]$XessSdkRoot = $env:XESS_SDK_ROOT,
    [string]$OutDir = "",
    [string]$BuildDir = "",
    [string]$CMake = "",
    [string]$Dxc = "",
    [string]$Adapter = "",
    [ValidateSet(1, 3)][int]$Mode = 3,
    [ValidateRange(2, 2147483647)][int]$Width = 640,
    [ValidateRange(2, 2147483647)][int]$Height = 360,
    [ValidateRange(1, 2147483647)][int]$Frames = 2,
    [ValidateRange(0, 2147483647)][int]$Warmup = 4,
    [string]$Velocity = "12,4",
    [ValidateSet("standard", "smooth")][string]$Scene = "standard",
    [switch]$Static,
    [ValidateRange(1, 2147483647)][int]$TensorDispatches = 76,
    [ValidateRange(0, 2147483647)][int]$TensorBudgetMB = 0
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "tools.ps1")

if ([string]::IsNullOrWhiteSpace($XessSdkRoot)) {
    throw "Pass -XessSdkRoot or set XESS_SDK_ROOT to the XeSS SDK 3.0.2 root."
}

$sdkRoot = Resolve-XefgFullPath $XessSdkRoot
if (-not (Test-Path -LiteralPath $sdkRoot -PathType Container)) {
    throw "XessSdkRoot does not exist or is not a directory: $sdkRoot"
}
foreach ($relative in @(
    "inc\xess_fg\xefg_swapchain_d3d12.h",
    "inc\xell\xell_d3d12.h",
    "bin\libxess_fg.dll",
    "bin\libxell.dll"
)) {
    $required = Join-Path $sdkRoot $relative
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "XeSS SDK prerequisite is missing: $required"
    }
}

Assert-XefgDimensions -Width $Width -Height $Height
if ($Velocity -notmatch '^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*,\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$') {
    throw "Velocity must be two comma-separated numbers, for example '12,4' (got '$Velocity')."
}

$tempRoot = [System.IO.Path]::GetTempPath()
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
    $OutDir = Join-Path $tempRoot ("xefg-sdk-capture-" + $stamp + "-" + $suffix)
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $tempRoot "xefg-sdk-capture-build"
}

$outDirPath = Resolve-XefgFullPath $OutDir
$buildDirPath = Resolve-XefgFullPath $BuildDir
Assert-XefgOutputDirectoryAvailable -Path $outDirPath

if (Test-Path -LiteralPath $buildDirPath) {
    if (-not (Test-Path -LiteralPath $buildDirPath -PathType Container)) {
        throw "BuildDir exists and is not a directory: $buildDirPath"
    }
}
if ($outDirPath.TrimEnd('\', '/') -eq $buildDirPath.TrimEnd('\', '/')) {
    throw "OutDir and BuildDir must be different directories."
}

$cmakeSource = Join-Path $PSScriptRoot "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeSource -PathType Leaf)) {
    throw "Local CMakeLists.txt is missing from $PSScriptRoot"
}

# Resolve every configured tool before building so a bad override fails early.
$cmakePath = Find-XefgCMake -Explicit $CMake
$dxcPath = Find-XefgDxc -Explicit $Dxc

$motionWidth = [int]($Width / 2)
$motionHeight = [int]($Height / 2)
$tensorBudget = Get-XefgTensorBudgetMB -Width $Width -Height $Height -OverrideMB $TensorBudgetMB
$adapterFilter = $Adapter.Trim()

Write-Host "Configuring XefgProbe..."
& $cmakePath -S $PSScriptRoot -B $buildDirPath -G "Visual Studio 17 2022" -A x64 "-DXESS_SDK_ROOT=$sdkRoot"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE (build directory: $buildDirPath)"
}

Write-Host "Building XefgProbe..."
& $cmakePath --build $buildDirPath --config Release --target XefgProbe --parallel
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE (build directory: $buildDirPath)"
}

$exe = $null
foreach ($candidate in @(
    (Join-Path $buildDirPath "Release\XefgProbe.exe"),
    (Join-Path $buildDirPath "XefgProbe.exe")
)) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $exe = $candidate
        break
    }
}
if (-not $exe) {
    $found = Get-ChildItem -LiteralPath $buildDirPath -Filter XefgProbe.exe -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($found) { $exe = $found.FullName }
}
if (-not $exe) {
    throw "XefgProbe.exe was not produced under $buildDirPath"
}

# Re-check immediately before creation so the caller cannot silently overwrite
# a capture that appeared while the project was building.
Assert-XefgOutputDirectoryAvailable -Path $outDirPath
[void][System.IO.Directory]::CreateDirectory($outDirPath)

$captureArguments = @(
    "--xess-dir", (Join-Path $sdkRoot "bin"),
    "--out", $outDirPath,
    "--width", [string]$Width,
    "--height", [string]$Height,
    "--mv-width", [string]$motionWidth,
    "--mv-height", [string]$motionHeight,
    "--frames", [string]$Frames,
    "--warmup", [string]$Warmup,
    "--velocity", $Velocity,
    "--scene", $Scene,
    "--tensors",
    "--tensor-budget-mb", [string]$tensorBudget,
    "--tensor-dispatches", [string]$TensorDispatches,
    "--no-model-dump"
)
if ($Mode -eq 3) {
    $captureArguments += @("--ui", "on", "--ui-mode", "hudless-uitexture")
}
else {
    $captureArguments += @("--ui", "off", "--ui-mode", "none")
}
if ($adapterFilter) {
    $captureArguments += @("--adapter", $adapterFilter)
}
if ($Static) {
    $captureArguments += "--static"
}

Write-Host ""
Write-Host "Capture configuration:"
Write-Host "  adapter:  $(if ($adapterFilter) { $adapterFilter } else { '<first hardware>' })"
Write-Host "  mode:     $Mode"
Write-Host "  output:   ${Width}x${Height}, motion/depth ${motionWidth}x${motionHeight}"
Write-Host "  scene:    $Scene$(if ($Static) { ', static' } else { '' })"
Write-Host "  frames:   $Frames after $Warmup warmup frame(s)"
Write-Host "  tensors:  first $TensorDispatches dispatch(es), budget $tensorBudget MB"
Write-Host "  out:      $outDirPath"
Write-Host ""

& $exe @captureArguments
if ($LASTEXITCODE -ne 0) {
    throw "XefgProbe failed with exit code $LASTEXITCODE; inspect $outDirPath"
}

Write-Host ""
Write-Host "Checking capture metadata and requested tensor frames..."
Assert-XefgCapture -RunDir $outDirPath -TensorDispatches $TensorDispatches -Width $Width -Height $Height

Write-Host "Disassembling captured programs..."
try {
    & (Join-Path $PSScriptRoot "disasm.ps1") -RunDir $outDirPath -Dxc $dxcPath
}
catch {
    throw "Disassembly failed: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "Run written to $outDirPath"
$completeFrames = [Math]::Floor($TensorDispatches / 38)
if ($completeFrames -ge 1) {
    Write-Host "Tensor-complete trace frames: 1..$completeFrames (38 dispatches each)."
}
else {
    Write-Host "Tensor snapshots cover dispatch 1..$TensorDispatches; no complete 38-dispatch frame was requested."
}
Write-Host "Export with: xefg-export --runs `"$outDirPath`" --out <new-package-directory>"
