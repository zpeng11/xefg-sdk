#requires -Version 5.1
# Basic, dependency-free tests for the capture scripts. These cover argument
# validation, output-directory safety, trace/tensor validation, and disassembly
# failure/overwrite behavior. They never start a GPU capture or build the project.

$ErrorActionPreference = "Stop"
$captureRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $captureRoot "tools.ps1")

$script:Passed = 0
$script:Failed = 0
$script:TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("xefg-script-tests-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
[void][System.IO.Directory]::CreateDirectory($script:TempRoot)

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Throws {
    param([scriptblock]$Action, [string]$Pattern = "")
    $message = ""
    $threw = $false
    try {
        & $Action
    }
    catch {
        $threw = $true
        $message = $_.Exception.Message
    }
    if (-not $threw) { throw "Expected an error matching '$Pattern'." }
    if ($Pattern -and $message -notmatch $Pattern) {
        throw "Error did not match '$Pattern': $message"
    }
}

function Test-Case {
    param([string]$Name, [scriptblock]$Body)
    try {
        & $Body
        $script:Passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:Failed++
        Write-Host "FAIL $Name"
        Write-Host "     $($_.Exception.Message)"
    }
}

function Invoke-TestProcess {
    param([string]$Executable, [string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Text = [string]::Join("`n", @($output | ForEach-Object { [string]$_ }))
    }
}

function New-TestFile {
    param([string]$Path, [string]$Content = "x")
    $parent = Split-Path -Parent $Path
    [void][System.IO.Directory]::CreateDirectory($parent)
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

function New-SyntheticCapture {
    param([string]$Root, [int]$Dispatches = 1)
    $trace = Join-Path $Root "trace"
    [void][System.IO.Directory]::CreateDirectory((Join-Path $trace "tensors"))
    [void][System.IO.Directory]::CreateDirectory((Join-Path $trace "shaders"))
    New-TestFile (Join-Path $trace "shaders\d1-0123456789abcdef.dxil") "dxil"
    $meta = '{"tensor_wait":true,"tensor_wait_error":"","trace_error":"","tensor_files":' + $Dispatches + ',"configuration":{"tensors":true,"width":64,"height":36}}'
    New-TestFile (Join-Path $trace "meta.json") $meta

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('{"type":"program","key":"d1"}')
    for ($id = 1; $id -le $Dispatches; $id++) {
        $name = "d{0:D4}_t0_in.bin" -f $id
        New-TestFile (Join-Path $trace ("tensors\" + $name)) "tensor"
        $lines.Add('{"type":"dispatch","dispatch":' + $id + ',"resources":[{"file":"' + $name + '"}]}')
    }
    $lines.Add('{"type":"pipeline","pso":"0x1"}')
    New-TestFile (Join-Path $trace "trace.jsonl") (($lines -join "`n") + "`n")
}

function New-ModeCapture {
    param([string]$Root, [int]$D38Mode, [int]$MetaMode = $D38Mode)
    # One bounded frame with the 38 captured dispatches. Only d38 carries the
    # composite constants, and only d1 has a tensor file we ask the checker for.
    $trace = Join-Path $Root "trace"
    [void][System.IO.Directory]::CreateDirectory((Join-Path $trace "tensors"))
    [void][System.IO.Directory]::CreateDirectory((Join-Path $trace "shaders"))
    New-TestFile (Join-Path $trace "shaders\d1-0123456789abcdef.dxil") "dxil"
    New-TestFile (Join-Path $trace "tensors\d0001_t0_in.bin") "tensor"
    $meta = '{"tensor_wait":true,"tensor_wait_error":"","trace_error":"","tensor_files":1,"configuration":{"tensors":true,"width":64,"height":36,"ui_mode":' + $MetaMode + '}}'
    New-TestFile (Join-Path $trace "meta.json") $meta

    $lines = New-Object System.Collections.Generic.List[string]
    for ($id = 1; $id -le 38; $id++) {
        $constants = '[]'
        if ($id -eq 38) {
            $values = @(0) * 15
            $values[14] = $D38Mode
            $constants = '[{"parameter":0,"values":[' + ($values -join ',') + ']}]'
        }
        $resources = '[]'
        if ($id -eq 1) { $resources = '[{"file":"d0001_t0_in.bin"}]' }
        $lines.Add('{"type":"dispatch","dispatch":' + $id + ',"frame":1,"constants":' + $constants + ',"resources":' + $resources + '}')
    }
    New-TestFile (Join-Path $trace "trace.jsonl") (($lines -join "`n") + "`n")
}

try {
    Test-Case "PowerShell parses all capture scripts" {
        foreach ($name in @("tools.ps1", "run.ps1", "disasm.ps1")) {
            $tokens = $null
            $errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseFile(
                (Join-Path $captureRoot $name), [ref]$tokens, [ref]$errors)
            if ($errors.Count -gt 0) {
                throw "$name parse error: $($errors[0].Message)"
            }
        }
    }

    Test-Case "Tensor budget defaults are bounded and scale from 4K" {
        Assert-True ((Get-XefgTensorBudgetMB -Width 640 -Height 360) -eq 512) "640x360 floor should be 512 MB."
        Assert-True ((Get-XefgTensorBudgetMB -Width 1920 -Height 1080) -eq 1536) "1080p budget should be 1536 MB."
        Assert-True ((Get-XefgTensorBudgetMB -Width 3840 -Height 2160) -eq 6144) "4K budget should be 6144 MB."
        Assert-True ((Get-XefgTensorBudgetMB -Width 640 -Height 360 -OverrideMB 100) -eq 100) "Explicit budget should win."
    }

    Test-Case "Output dimensions must be positive and even" {
        Assert-XefgDimensions -Width 640 -Height 360
        Assert-Throws { Assert-XefgDimensions -Width 641 -Height 360 } "even"
        Assert-Throws { Assert-XefgDimensions -Width 640 -Height 0 } "positive"
    }

    Test-Case "OutDir must be absent or empty" {
        $missing = Join-Path $script:TempRoot "out-missing"
        Assert-XefgOutputDirectoryAvailable -Path $missing

        $empty = Join-Path $script:TempRoot "out-empty"
        [void][System.IO.Directory]::CreateDirectory($empty)
        Assert-XefgOutputDirectoryAvailable -Path $empty

        $full = Join-Path $script:TempRoot "out-full"
        [void][System.IO.Directory]::CreateDirectory($full)
        New-TestFile (Join-Path $full "file.txt")
        Assert-Throws { Assert-XefgOutputDirectoryAvailable -Path $full } "absent or empty"
    }

    Test-Case "Complete synthetic trace passes the capture checks" {
        $run = Join-Path $script:TempRoot "run-good"
        New-SyntheticCapture -Root $run -Dispatches 2
        Assert-XefgCapture -RunDir $run -TensorDispatches 2 -Width 64 -Height 36
    }

    Test-Case "Mode verification reads d38 c[14] from the trace, not meta" {
        $run = Join-Path $script:TempRoot "run-mode4"
        New-ModeCapture -Root $run -D38Mode 4 -MetaMode 1
        Assert-XefgCapture -RunDir $run -TensorDispatches 1 -Mode 4
        Assert-Throws { Assert-XefgCapture -RunDir $run -TensorDispatches 1 -Mode 3 } "d38 c\[14\]"
    }

    Test-Case "Missing tensor file is rejected" {
        $run = Join-Path $script:TempRoot "run-missing-tensor"
        New-SyntheticCapture -Root $run -Dispatches 2
        Remove-Item -LiteralPath (Join-Path $run "trace\tensors\d0002_t0_in.bin") -Force
        Assert-Throws { Assert-XefgCapture -RunDir $run -TensorDispatches 2 } "missing"
    }

    Test-Case "Missing dispatch row is rejected" {
        $run = Join-Path $script:TempRoot "run-missing-row"
        New-SyntheticCapture -Root $run -Dispatches 1
        Assert-Throws { Assert-XefgCapture -RunDir $run -TensorDispatches 2 } "missing dispatch row"
    }

    Test-Case "Invalid trace JSON is rejected" {
        $run = Join-Path $script:TempRoot "run-bad-json"
        New-SyntheticCapture -Root $run -Dispatches 1
        Add-Content -LiteralPath (Join-Path $run "trace\trace.jsonl") -Value "{not json"
        Assert-Throws { Assert-XefgCapture -RunDir $run -TensorDispatches 1 } "not valid JSON"
    }

    Test-Case "Disassembly names, reruns, modified-output rejection, and DXC failure" {
        $run = Join-Path $script:TempRoot "run-disasm"
        [void][System.IO.Directory]::CreateDirectory((Join-Path $run "trace\shaders"))
        New-TestFile (Join-Path $run "trace\shaders\d1-0123456789abcdef.dxil") "dxil"

        $goodStub = Join-Path $script:TempRoot "dxc-good.cmd"
        New-TestFile $goodStub "@echo off`r`necho target triple = `"dxil-ms-dx`"`r`necho define void @sample`r`nexit /b 0`r`n"

        $disasm = Join-Path $captureRoot "disasm.ps1"
        & $disasm -RunDir $run -Dxc $goodStub
        $target = Join-Path $run "trace\disasm\d1-0123456789abcdef.ll"
        Assert-True (Test-Path -LiteralPath $target -PathType Leaf) "Expected .ll output was not produced."
        $text = [System.IO.File]::ReadAllText($target)
        Assert-True ($text -match 'target triple') "Disassembly does not contain LLVM IR."

        & $disasm -RunDir $run -Dxc $goodStub
        Add-Content -LiteralPath $target -Value "modified"
        Assert-Throws { & $disasm -RunDir $run -Dxc $goodStub } "modified disassembly"

        $badRun = Join-Path $script:TempRoot "run-disasm-bad"
        [void][System.IO.Directory]::CreateDirectory((Join-Path $badRun "trace\shaders"))
        New-TestFile (Join-Path $badRun "trace\shaders\bad.dxil") "dxil"
        $badStub = Join-Path $script:TempRoot "dxc-bad.cmd"
        New-TestFile $badStub "@echo off`r`necho dxc failed 1>&2`r`nexit /b 1`r`n"
        Assert-Throws { & $disasm -RunDir $badRun -Dxc $badStub } "dxc -dumpbin failed"
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $badRun "trace\disasm\bad.ll"))) "Failed DXC must not leave a disassembly file."
    }

    Test-Case "run.ps1 rejects a missing SDK before building" {
        $powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
        $runScript = Join-Path $captureRoot "run.ps1"
        $result = Invoke-TestProcess $powershell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runScript,
            "-XessSdkRoot", (Join-Path $script:TempRoot "missing-sdk"),
            "-OutDir", (Join-Path $script:TempRoot "out-run-missing-sdk"))
        Assert-True ($result.ExitCode -ne 0) "run.ps1 should fail for a missing SDK."
        Assert-True ($result.Text -match "XessSdkRoot") "Missing-SDK error was not reported."
    }

    Test-Case "run.ps1 rejects uneven dimensions before building" {
        $powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
        $runScript = Join-Path $captureRoot "run.ps1"
        $sdk = Join-Path $script:TempRoot "fake-sdk"
        New-TestFile (Join-Path $sdk "inc\xess_fg\xefg_swapchain_d3d12.h")
        New-TestFile (Join-Path $sdk "inc\xell\xell_d3d12.h")
        New-TestFile (Join-Path $sdk "bin\libxess_fg.dll")
        New-TestFile (Join-Path $sdk "bin\libxell.dll")
        $result = Invoke-TestProcess $powershell @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runScript,
            "-XessSdkRoot", $sdk, "-OutDir", (Join-Path $script:TempRoot "out-run-odd"),
            "-BuildDir", (Join-Path $script:TempRoot "build-run-odd"),
            "-Width", "641", "-Height", "360")
        Assert-True ($result.ExitCode -ne 0) "run.ps1 should fail for an uneven width."
        Assert-True ($result.Text -match "must be even") "Uneven-width error was not reported."
    }
}
finally {
    Remove-Item -LiteralPath $script:TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Passed: $script:Passed  Failed: $script:Failed"
if ($script:Failed -ne 0) {
    exit 1
}
