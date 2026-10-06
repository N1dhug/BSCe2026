param(
    [string]$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..")),
    [string]$RunDir = "files/output/bm_1/runs/estimation/h2_precise_development_pc_v1",
    [ValidateRange(15, 900)][int]$Seconds = 900,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

# Source must remain fixed while any model process uses this checkout.
$activeModels = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -like "python*.exe" -or $_.Name -eq "py.exe") -and
    $_.CommandLine -like "*src.exe.run_model*"
})
if ($activeModels.Count -gt 0) {
    throw "A model run is active. Preserve its source; wait before starting this pilot."
}
$trackedChanges = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "Git status failed." }
if ($trackedChanges.Count -gt 0) { throw "Use clean committed source before the pilot." }

$configPath = "files/output/bm_1/runs/estimation/h2_precise_contract_pc_v1.json"
if (-not (Test-Path -LiteralPath $configPath)) {
    python -B -m src.exe.run_model.bm_1_h2_precise config --output $configPath
    if ($LASTEXITCODE -ne 0) { throw "H2 contract template creation failed." }
}

$arguments = @("-B", "-m", "src.exe.run_model.bm_1_h2_precise", "development-pilot",
    "--run-dir", $RunDir, "--workers", "8", "--seconds", "$Seconds")
if ($Resume) { $arguments += "--resume" }
python @arguments
if ($LASTEXITCODE -ne 0) { throw "H2 pilot failed. Preserve report, checkpoints and logs." }

Write-Host "Pilot report: $RunDir/pilot_report.json"
Write-Host "Historical contract template: $configPath"
Write-Host "Common H budget: files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json"
Write-Host "This is synthetic runtime evidence, not a historical fit or H2/H3 acceptance."
