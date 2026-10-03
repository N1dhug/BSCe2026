# Run under Windows PowerShell 5.1. Real Git; stub only the expensive Python job.
$ErrorActionPreference = "Stop"
$launcherSource = Join-Path $PSScriptRoot "../root/f4/run_g2b_night.ps1"
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("bm night " + [guid]::NewGuid())
$testCheckout = Join-Path $testRoot ("checkout " + [char]0x00e6)
$testOrigin = Join-Path $testRoot "origin.git"
$testOriginalLocation = Get-Location

function Test-Git {
    $result = & git @args
    if ($LASTEXITCODE -ne 0) { throw "Fixture Git failed: $args" }
    $result
}
function Assert-Night($condition, $message) {
    if (!$condition) { throw $message }
}
function Argument-Value($arguments, $name) {
    $index = [array]::IndexOf($arguments, $name)
    Assert-Night ($index -ge 0) "Missing argument: $name"
    $arguments[$index + 1]
}

$global:nightLauncherCalls = @()
$global:nightLauncherExit = 0
function global:python {
    $call = @($args)
    $global:nightLauncherCalls += ,$call
    $directory = Argument-Value $call "--run-dir"
    $manifest = Join-Path $directory "manifest.json"
    if (!(Test-Path -LiteralPath $manifest)) {
        $revision = Test-Git rev-parse HEAD
        @{source = @{source_commit = $revision}} | ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath $manifest -Encoding UTF8
    }
    Write-Output "Stub Python reached with recorded CLI arguments."
    $global:LASTEXITCODE = $global:nightLauncherExit
}

try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    Test-Git init --bare --initial-branch=main $testOrigin | Out-Null
    Test-Git clone $testOrigin $testCheckout | Out-Null
    Set-Location -LiteralPath $testCheckout
    Test-Git config user.name "Launcher test"
    Test-Git config user.email "launcher-test@example.invalid"
    New-Item -ItemType Directory -Force -Path root/f4, src/validation/bm_1 | Out-Null
    Copy-Item -LiteralPath $launcherSource -Destination root/f4/run_g2b_night.ps1
    "# Fixture module" | Set-Content src/validation/bm_1/joint_night.py
    "files/output/" | Set-Content .gitignore
    Test-Git add .
    Test-Git commit -m "Fixture source" | Out-Null
    Test-Git push origin main
    $originalSource = Test-Git rev-parse HEAD

    $output = "files/output/bm_1/runs/estimation"
    $g2a = "$output/f4_g_g2a_rectangular_v2_20261003_135421"
    New-Item -ItemType Directory -Force -Path $g2a | Out-Null
    "{}" | Set-Content "$g2a/manifest.json"
    "stub" | Set-Content "$output/f4_g_g2a_continued_7d_v1.json.gz"

    & ./root/f4/run_g2b_night.ps1 -LimitChunks 1
    Assert-Night ($global:nightLauncherCalls.Count -eq 1) "Pilot did not launch exactly once."
    $first = $global:nightLauncherCalls[0]
    Assert-Night ((Argument-Value $first "--max-hours") -eq "8") "Wrong default budget."
    Assert-Night ((Argument-Value $first "--workers") -eq "8") "Wrong worker count."
    Assert-Night ((Argument-Value $first "--blas-threads") -eq "1") "Wrong BLAS limit."
    Assert-Night ((Argument-Value $first "--chunk-size") -eq "64") "Wrong checkpoint block."
    Assert-Night ((Argument-Value $first "--limit-chunks") -eq "1") "Pilot limit was lost."
    Assert-Night ((Test-Git branch --show-current) -eq "night/g2b-v2") "Night branch not selected."
    $runDirectory = Argument-Value $first "--run-dir"

    Test-Git switch main
    "Daytime work" | Set-Content README.md
    Test-Git add README.md
    Test-Git commit -m "Advance daytime main" | Out-Null
    Test-Git push origin main
    & ./root/f4/run_g2b_night.ps1 -MaxHours 8
    Assert-Night ($global:nightLauncherCalls.Count -eq 2) "Resume did not launch exactly once."
    Assert-Night ((Test-Git rev-parse HEAD) -eq $originalSource) "Resume changed the pinned source."
    $second = $global:nightLauncherCalls[1]
    Assert-Night ((Argument-Value $second "--run-dir") -eq $runDirectory) "Resume changed run directory."
    Assert-Night (!($second -contains "--limit-chunks")) "Night run retained pilot block limit."
    Assert-Night (@(Get-ChildItem -LiteralPath $runDirectory -Filter "night_*.txt").Count -eq 2) "Separate logs missing."

    $global:nightLauncherExit = 7
    $failed = $false
    try { & ./root/f4/run_g2b_night.ps1 } catch {
        $failed = $_.Exception.Message -like "Koerselsfejl*"
    }
    Assert-Night $failed "Python failure did not stop the launcher."
    Assert-Night ($global:nightLauncherCalls.Count -eq 3) "Failure check did not reach Python."
    Assert-Night (Test-Path -LiteralPath "$runDirectory/manifest.json") "Failure removed saved manifest."

    "# Local change" | Add-Content root/f4/run_g2b_night.ps1
    $failed = $false
    try { & ./root/f4/run_g2b_night.ps1 } catch {
        $failed = $_.Exception.Message -like "Lokale aendringer*"
    }
    Assert-Night $failed "Dirty checkout was accepted."
    Assert-Night ($global:nightLauncherCalls.Count -eq 3) "Dirty checkout launched Python."
    Write-Host "Windows launcher: first block, pinned resume, logs, Python errors and dirty-source guard passed."
} finally {
    Set-Location -LiteralPath $testOriginalLocation
    Remove-Item Function:\python -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
