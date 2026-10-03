# Start or resume the same G2b-v2 study; creates a branch, never a worktree.
[CmdletBinding()]
param(
    [ValidateRange(0.000001, 168)][double]$MaxHours = 8,
    [ValidateRange(1, 256)][int]$Workers = 8,
    [ValidateRange(1, 1000000)][int]$ChunkSize = 64,
    [ValidateRange(0, 1000000)][int]$LimitChunks = 0
)

& {
    $ErrorActionPreference = "Stop"
    $nightRepo = & git -C $PSScriptRoot rev-parse --show-toplevel
    if ($LASTEXITCODE -ne 0) { throw "Repository could not be found." }
    Set-Location -LiteralPath $nightRepo
    function Night-Git {
        param([string[]]$GitArgs)
        $nightResult = & git @GitArgs
        if ($LASTEXITCODE -ne 0) { throw "Git fejlede: $($GitArgs -join ' ')" }
        $nightResult
    }
    if (Night-Git @("status", "--porcelain")) {
        throw "Lokale aendringer fundet; behold dem."
    }
    Night-Git @("fetch", "--prune", "origin")
    $nightRoot = "files/output/bm_1/runs/estimation"
    $nightOld = "$nightRoot/f4_g_g2a_continued_7d_v1.json.gz"
    $nightG2a = "$nightRoot/f4_g_g2a_rectangular_v2_20261003_135421"
    $nightDir = "$nightRoot/f4_g_g2b_rectangular_v2_nights"
    $nightBranch = "night/g2b-v2"
    if (!(Test-Path -LiteralPath $nightOld) -or
        !(Test-Path -LiteralPath "$nightG2a/manifest.json")) {
        throw "Den komplette G2a-dependency mangler; behold alle resultater."
    }
    if (Test-Path -LiteralPath "$nightDir/manifest.json") {
        $nightSource = (Get-Content "$nightDir/manifest.json" -Raw |
            ConvertFrom-Json).source.source_commit
    } else {
        $nightSource = Night-Git @("rev-parse", "origin/main")
    }
    Night-Git @("cat-file", "-e", "${nightSource}:src/validation/bm_1/joint_night.py")
    $nightBranchSha = Night-Git @("for-each-ref", "--format=%(objectname)", "refs/heads/$nightBranch")
    if ($nightBranchSha) {
        if ($nightBranchSha -ne $nightSource) { throw "Nat-branchen har en anden source; behold den." }
    } else {
        Night-Git @("branch", $nightBranch, $nightSource)
    }
    Night-Git @("switch", $nightBranch)
    if ((Night-Git @("rev-parse", "HEAD")) -ne $nightSource) { throw "Forkert nat-source." }
    New-Item -ItemType Directory -Force -Path $nightDir | Out-Null
    $nightLog = "$nightDir/night_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff').txt"
    $nightExtra = @()
    if ($LimitChunks -gt 0) { $nightExtra += @("--limit-chunks", "$LimitChunks") }
    # Continue here preserves native Python stderr in Windows PowerShell's tee.
    $ErrorActionPreference = "Continue"
    python -B -m src.exe.run_model.bm_1.estimate joint-bank-rectangular-robustness `
        --g2a-report $nightOld --g2a-run-dir $nightG2a --run-dir $nightDir `
        --workers $Workers --chunk-size $ChunkSize --max-hours $MaxHours --blas-threads 1 @nightExtra `
        2>&1 | Tee-Object -FilePath $nightLog
    $nightExit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Write-Host "Summary: $nightDir/summary.json"
    Write-Host "Log: $nightLog"
    if ($nightExit -ne 0) { throw "Koerselsfejl eller numerisk afvisning; behold hele run-mappen." }
}
