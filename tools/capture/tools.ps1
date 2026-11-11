#requires -Version 5.1
# Shared helpers for run.ps1 and disasm.ps1. Kept dependency-free so the
# capture scripts remain usable from a plain Windows PowerShell 5.1 prompt.

function Resolve-XefgFullPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$BasePath = ""
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "A non-empty path is required."
    }
    if ([string]::IsNullOrWhiteSpace($BasePath)) {
        $BasePath = (Get-Location).ProviderPath
    }

    try {
        if ([System.IO.Path]::IsPathRooted($Path)) {
            return [System.IO.Path]::GetFullPath($Path)
        }
        return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
    }
    catch {
        throw "Invalid path '$Path': $($_.Exception.Message)"
    }
}

function Get-XefgVswherePath {
    $candidates = @()
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return ""
}

function Find-XefgCMake {
    [CmdletBinding()]
    param([string]$Explicit = "")

    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        $path = Resolve-XefgFullPath $Explicit
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "CMake executable not found: $path"
        }
        return $path
    }

    $command = Get-Command cmake.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandType -eq "Application" } |
        Select-Object -First 1
    if ($command -and $command.Path) {
        return $command.Path
    }

    $vswhere = Get-XefgVswherePath
    if ($vswhere) {
        $found = @(& $vswhere -latest -products '*' -find "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" 2>$null)
        foreach ($candidate in $found) {
            if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                return $candidate
            }
        }
    }

    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $candidate = Join-Path $root "CMake\bin\cmake.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "cmake.exe not found (checked PATH, vswhere, and standard CMake install roots); pass -CMake <path>."
}

function Find-XefgDxc {
    [CmdletBinding()]
    param([string]$Explicit = "")

    if (-not [string]::IsNullOrWhiteSpace($Explicit)) {
        $path = Resolve-XefgFullPath $Explicit
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "DXC executable not found: $path"
        }
        return $path
    }

    $command = Get-Command dxc.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandType -eq "Application" } |
        Select-Object -First 1
    if ($command -and $command.Path) {
        return $command.Path
    }

    $kitRoots = @()
    if ($env:WindowsSdkDir) {
        $kitRoots += $env:WindowsSdkDir
    }
    if (${env:ProgramFiles(x86)}) {
        $kitRoots += (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10")
    }
    if ($env:ProgramFiles) {
        $kitRoots += (Join-Path $env:ProgramFiles "Windows Kits\10")
    }

    foreach ($kit in $kitRoots) {
        $bin = Join-Path $kit "bin"
        if (-not (Test-Path -LiteralPath $bin -PathType Container)) { continue }
        $candidate = Get-ChildItem -Path (Join-Path $bin "*\x64\dxc.exe") -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    $vswhere = Get-XefgVswherePath
    if ($vswhere) {
        $found = @(& $vswhere -latest -products '*' -find "VC\Tools\MSVC\**\bin\Hostx64\x64\dxc.exe" 2>$null)
        foreach ($candidate in $found) {
            if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                return $candidate
            }
        }
    }

    throw "dxc.exe not found (checked PATH, Windows SDK, and vswhere); pass -Dxc <path>."
}

function Assert-XefgDimensions {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][int]$Width,
        [Parameter(Mandatory = $true)][int]$Height
    )

    foreach ($item in @(
        @{ Name = "Width"; Value = $Width },
        @{ Name = "Height"; Value = $Height }
    )) {
        if ($item.Value -le 0) {
            throw "$($item.Name) must be positive (got $($item.Value))."
        }
        if (($item.Value % 2) -ne 0) {
            throw "$($item.Name) must be even because motion/depth are half size (got $($item.Value))."
        }
    }
}

function Assert-XefgOutputDirectoryAvailable {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "OutDir exists and is not a directory: $Path"
    }
    $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop)
    if ($children.Count -gt 0) {
        throw "OutDir must be absent or empty; found $($children.Count) item(s): $Path"
    }
}

function Get-XefgTensorBudgetMB {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][int]$Width,
        [Parameter(Mandatory = $true)][int]$Height,
        [int]$OverrideMB = 0
    )

    if ($OverrideMB -gt 0) {
        return $OverrideMB
    }

    # 6144 MB retained both complete 4K frames in the baseline evidence. Scale
    # that budget by output pixels, with a 512 MB floor for small captures.
    $referencePixels = 3840.0 * 2160.0
    $pixels = [double]$Width * [double]$Height
    $budget = [int][Math]::Ceiling(6144.0 * $pixels / $referencePixels)
    if ($budget -lt 512) { $budget = 512 }
    if ($budget -gt 6144) { $budget = 6144 }
    return $budget
}

function Assert-XefgCapture {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RunDir,
        [Parameter(Mandatory = $true)][int]$TensorDispatches,
        [int]$Width = 0,
        [int]$Height = 0,
        [ValidateRange(0, 2147483647)][int]$Mode = 0
    )

    $traceDir = Join-Path $RunDir "trace"
    $metaPath = Join-Path $traceDir "meta.json"
    $tracePath = Join-Path $traceDir "trace.jsonl"
    $tensorDir = Join-Path $traceDir "tensors"
    $shaderDir = Join-Path $traceDir "shaders"

    if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
        throw "Capture is incomplete: trace/meta.json is missing under $RunDir"
    }
    if (-not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
        throw "Capture is incomplete: trace/trace.jsonl is missing under $RunDir"
    }
    if (-not (Test-Path -LiteralPath $tensorDir -PathType Container)) {
        throw "Capture is incomplete: trace/tensors/ is missing under $RunDir"
    }
    if (-not (Test-Path -LiteralPath $shaderDir -PathType Container)) {
        throw "Capture is incomplete: trace/shaders/ is missing under $RunDir"
    }

    $shaders = @(Get-ChildItem -LiteralPath $shaderDir -Filter *.dxil -File -ErrorAction Stop)
    if ($shaders.Count -eq 0) {
        throw "Capture is incomplete: no captured .dxil programs under $shaderDir"
    }

    try {
        $meta = Get-Content -LiteralPath $metaPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "trace/meta.json is not valid JSON: $($_.Exception.Message)"
    }
    if ($null -eq $meta) {
        throw "trace/meta.json is empty."
    }
    if (-not $meta.PSObject.Properties["tensor_wait"] -or $meta.tensor_wait -ne $true) {
        throw "trace/meta.json tensor_wait is not true; tensor dumps are not verified."
    }
    if (-not $meta.PSObject.Properties["trace_error"] -or -not [string]::IsNullOrEmpty([string]$meta.trace_error)) {
        throw "trace/meta.json trace_error is not empty: $($meta.trace_error)"
    }
    if ($meta.PSObject.Properties["tensor_wait_error"] -and -not [string]::IsNullOrEmpty([string]$meta.tensor_wait_error)) {
        throw "trace/meta.json tensor_wait_error is not empty: $($meta.tensor_wait_error)"
    }
    if ($meta.PSObject.Properties["configuration"] -and $null -ne $meta.configuration) {
        $configuration = $meta.configuration
        if ($configuration.PSObject.Properties["tensors"] -and $configuration.tensors -ne $true) {
            throw "trace/meta.json reports that tensor capture was disabled."
        }
        if ($Width -gt 0 -and $configuration.PSObject.Properties["width"] -and [int]$configuration.width -ne $Width) {
            throw "trace/meta.json width is $($configuration.width), expected $Width."
        }
        if ($Height -gt 0 -and $configuration.PSObject.Properties["height"] -and [int]$configuration.height -ne $Height) {
            throw "trace/meta.json height is $($configuration.height), expected $Height."
        }
    }
    if ($meta.PSObject.Properties["tensor_files"] -and [int]$meta.tensor_files -le 0) {
        throw "trace/meta.json reports no tensor files."
    }

    $rows = @{}
    $lineNumber = 0
    foreach ($line in (Get-Content -LiteralPath $tracePath -ErrorAction Stop)) {
        $lineNumber++
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try {
            $event = $line | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            throw "trace/trace.jsonl line $lineNumber is not valid JSON: $($_.Exception.Message)"
        }
        if ($null -eq $event -or -not $event.PSObject.Properties["type"] -or [string]$event.type -ne "dispatch") {
            # Program, pipeline, root-signature and parameter rows are allowed
            # to exist without tensors; only dispatch rows are checked below.
            continue
        }
        $dispatch = 0
        if (-not [int]::TryParse([string]$event.dispatch, [ref]$dispatch)) {
            throw "trace/trace.jsonl line $lineNumber has an invalid dispatch id."
        }
        if ($rows.ContainsKey($dispatch)) {
            throw "trace/trace.jsonl contains duplicate dispatch row $dispatch."
        }
        $rows[$dispatch] = $event
    }

    # The final composite's c[14] records the variant actually used. Read it
    # from the captured trace, never from meta.json, so an explicit mode cannot
    # be relabelled by the producer's metadata.
    if ($Mode -gt 0) {
        $verifiedFrames = 0
        foreach ($group in @($rows.Values | Group-Object { [int]$_.frame })) {
            $ordered = @($group.Group | Sort-Object { [int]$_.dispatch })
            if ($ordered.Count -lt 38) {
                continue
            }
            $d38 = $ordered[37]
            $values = $null
            foreach ($constant in @($d38.constants)) {
                if ($null -eq $constant -or -not $constant.PSObject.Properties["values"]) {
                    continue
                }
                $candidate = @($constant.values)
                if ($candidate.Count -ge 15) {
                    $values = $candidate
                    break
                }
            }
            if ($null -eq $values) {
                throw "trace d38 in frame $($group.Name) has no constant values for c[14]; cannot verify mode $Mode."
            }
            $actual = [int]$values[14]
            if ($actual -ne $Mode) {
                throw "trace d38 c[14]=$actual in frame $($group.Name) does not match requested mode $Mode."
            }
            $verifiedFrames++
        }
        if ($verifiedFrames -eq 0) {
            throw "trace has no complete 38-dispatch frame; cannot verify mode $Mode."
        }
    }

    $missingRows = New-Object System.Collections.Generic.List[string]
    $badTensors = New-Object System.Collections.Generic.List[string]
    for ($dispatch = 1; $dispatch -le $TensorDispatches; $dispatch++) {
        if (-not $rows.ContainsKey($dispatch)) {
            $missingRows.Add("d$dispatch")
            continue
        }

        $row = $rows[$dispatch]
        $resources = @()
        if ($row.PSObject.Properties["resources"] -and $null -ne $row.resources) {
            $resources = @($row.resources)
        }

        $recorded = 0
        foreach ($resource in $resources) {
            if ($null -eq $resource -or -not $resource.PSObject.Properties["file"]) { continue }
            $name = [string]$resource.file
            if ([string]::IsNullOrWhiteSpace($name)) { continue }
            $recorded++
            $path = Join-Path $tensorDir $name
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                $badTensors.Add("d${dispatch}: $name (missing)")
                continue
            }
            $item = Get-Item -LiteralPath $path -ErrorAction Stop
            if ($item.Length -le 0) {
                $badTensors.Add("d${dispatch}: $name (empty)")
            }
        }
        if ($recorded -eq 0) {
            $badTensors.Add("d${dispatch}: no tensor file recorded")
        }
    }

    if ($missingRows.Count -gt 0) {
        throw "Requested tensor frames are incomplete: missing dispatch row(s) $($missingRows -join ', ')."
    }
    if ($badTensors.Count -gt 0) {
        $preview = ($badTensors | Select-Object -First 5) -join "; "
        throw "Requested tensor frames are incomplete: $($badTensors.Count) tensor problem(s): $preview"
    }
}
