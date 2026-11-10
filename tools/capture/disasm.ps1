#requires -Version 5.1
<#
.SYNOPSIS
Disassembles the DXIL programs in an existing capture with Windows DXC.

.DESCRIPTION
Reads trace/shaders/*.dxil and writes trace/disasm/<shader-base-name>.ll.
DXC failures are fatal. An existing .ll file is never overwritten: if its text
matches the new DXC output it is accepted, otherwise the script fails and leaves
the file untouched.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$Dxc = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "tools.ps1")

$runDirPath = Resolve-XefgFullPath $RunDir
if (-not (Test-Path -LiteralPath $runDirPath -PathType Container)) {
    throw "RunDir does not exist or is not a directory: $runDirPath"
}

$shaderDir = Join-Path $runDirPath "trace\shaders"
if (-not (Test-Path -LiteralPath $shaderDir -PathType Container)) {
    throw "No trace/shaders/ directory under $runDirPath"
}

$shaders = @(Get-ChildItem -LiteralPath $shaderDir -Filter *.dxil -File -ErrorAction Stop |
    Sort-Object Name)
if ($shaders.Count -eq 0) {
    throw "No .dxil programs found under $shaderDir"
}

$dxcPath = Find-XefgDxc -Explicit $Dxc
$outDir = Join-Path $runDirPath "trace\disasm"
[void][System.IO.Directory]::CreateDirectory($outDir)

$written = 0
$unchanged = 0
$utf8 = New-Object System.Text.UTF8Encoding($false)

foreach ($shader in $shaders) {
    $targetName = [System.IO.Path]::GetFileNameWithoutExtension($shader.Name) + ".ll"
    $target = Join-Path $outDir $targetName

    # Native stderr becomes an ErrorRecord when redirected. Keep a DXC warning
    # from tearing down the pipeline before we can inspect $LASTEXITCODE.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $dxcPath -dumpbin $shader.FullName 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    $text = [string]::Join("`n", @($output | ForEach-Object { [string]$_ }))

    if ($exitCode -ne 0) {
        throw "dxc -dumpbin failed for $($shader.Name) with exit code $exitCode.`n$text"
    }
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "dxc produced no disassembly for $($shader.Name)"
    }
    if ($text -notmatch 'target triple\s*=' -and $text -notmatch '(?m)^define\s') {
        throw "dxc output for $($shader.Name) is not usable LLVM/DXIL disassembly."
    }

    $newText = $text.TrimEnd("`r", "`n") + "`n"
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            throw "Disassembly target exists and is not a file: $target"
        }
        $existing = [System.IO.File]::ReadAllText($target)
        $existingNormalized = $existing.Replace("`r`n", "`n").TrimEnd("`n")
        $newNormalized = $newText.Replace("`r`n", "`n").TrimEnd("`n")
        if ($existingNormalized -cne $newNormalized) {
            throw "Refusing to overwrite modified disassembly: $target"
        }
        $unchanged++
        continue
    }

    [System.IO.File]::WriteAllText($target, $newText, $utf8)
    if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
        (Get-Item -LiteralPath $target -ErrorAction Stop).Length -le 0) {
        throw "Failed to write usable disassembly: $target"
    }
    $written++
}

foreach ($shader in $shaders) {
    $targetName = [System.IO.Path]::GetFileNameWithoutExtension($shader.Name) + ".ll"
    $target = Join-Path $outDir $targetName
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        throw "Expected disassembly was not produced: $target"
    }
    if ((Get-Item -LiteralPath $target -ErrorAction Stop).Length -le 0) {
        throw "Expected disassembly is empty: $target"
    }
}

Write-Host "Disassembled $($shaders.Count) shader(s) into $outDir ($written written, $unchanged unchanged)"
