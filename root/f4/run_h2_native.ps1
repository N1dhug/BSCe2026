param(
    [ValidateRange(76, 900)][double]$Seconds = 900,
    [string]$RunDir = 'files/output/bm_1/runs/estimation/f4_h2_native_pc_20261006',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$active = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -like 'python*.exe' -or $_.Name -eq 'py.exe') -and
    $_.CommandLine -match 'src[./\\]exe[./\\]run_model'
})
if ($active.Count -gt 0) {
    throw 'En modelkørsel er aktiv. Bevar dens checkout; kør F4-H2 efter den er stoppet.'
}
$changes = @(git status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) { throw 'Git status fejlede.' }
if ($changes.Count -gt 0) {
    throw 'Tracked ændringer skal gemmes før source kan fastlåses.'
}
$env:OPENBLAS_NUM_THREADS = '1'
$env:OPENBLAS_DEFAULT_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$directory = $RunDir
$parent = Split-Path -Parent $directory
$stem = (Split-Path -Leaf $directory) -replace '_pc_', '_pc_review_'
$archive = Join-Path $parent "$stem.zip"
if ($Resume) {
    $archive = Join-Path $parent "$($stem)_resume_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
} elseif (Test-Path $archive) {
    throw "Bevar eksisterende arkiv: $archive"
}
$arguments = @('-B', '-m', 'src.exe.run_model.bm_1_h2_native', '--run-dir', $directory, '--seconds', "$Seconds")
if ($Resume) { $arguments += '--resume' }
python @arguments
$result = $LASTEXITCODE
if ($result -ne 0 -and $result -ne 3) {
    throw "F4-H2 native kontrol fejlede (exit $result). Bevar rapport/failure; ingen automatisk gentagelse."
}
Compress-Archive -Path $directory, 'files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json' -DestinationPath $archive
Write-Host "Returnér $archive til råreview. Exit=$result; pause er ikke accept."
