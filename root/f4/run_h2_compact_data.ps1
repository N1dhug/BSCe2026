param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [ValidateRange(76,300)][double]$Seconds=180
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path $RepoRoot).Path
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$snapshot=Join-Path ([System.IO.Path]::GetTempPath()) ('f4_h2_data_'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $snapshot | Out-Null
$sourceZip=Join-Path $snapshot 'source_export.zip'
# Export immutable files without changing the user's active branch/worktree.
git -C $RepoRoot archive --format=zip --output=$sourceZip $SourceCommit src root/f4 files/output/bm_1/runs/estimation/f4_h2_native_budget_seed_20261006.json
if ($LASTEXITCODE -ne 0) { throw 'Git source-export fejlede; ingen datakørsel startet.' }
$sourceRoot=Join-Path $snapshot 'code'
Expand-Archive -LiteralPath $sourceZip -DestinationPath $sourceRoot
$output=Join-Path $RepoRoot "files/output/bm_1/runs/estimation/f4_h2_compact_data_pc_$stamp"
$campaign=Join-Path $RepoRoot 'files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json'
$inputBook=Join-Path $RepoRoot 'files/input/data/input.xlsx'
$priceBook=Join-Path $RepoRoot 'files/input/data/prices.xlsx'
foreach ($book in @($inputBook,$priceBook)) {
    if (-not (Test-Path -LiteralPath $book)) { throw "Mangler eksisterende workbook: $book" }
}
python -B (Join-Path $sourceRoot 'root/f4/prepare_h2_compact_data.py') --input-workbook $inputBook --price-workbook $priceBook --output $output --campaign $campaign --source-commit $SourceCommit --max-seconds $Seconds
if ($LASTEXITCODE -ne 0) {
    throw "F4-H2 dataforberedelse stoppede. Bevar $output og ledgeren; ingen automatisk gentagelse."
}
Write-Host "Send review-ZIP: $($output)_review.zip"
