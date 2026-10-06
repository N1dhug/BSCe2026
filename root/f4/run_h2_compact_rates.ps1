param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [ValidateRange(76,300)][double]$Seconds=300
)
$ErrorActionPreference='Stop'
$launchWatch=[System.Diagnostics.Stopwatch]::StartNew()
$RepoRoot=(Resolve-Path $RepoRoot).Path
$snapshot=Join-Path ([System.IO.Path]::GetTempPath()) ('f4_h2_rates_'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $snapshot | Out-Null
$sourceZip=Join-Path $snapshot 'source_export.zip'
$packetBase='files/output/bm_1/runs/estimation'
# Immutable source/data export; no active branch/worktree change and no installs.
git -C $RepoRoot archive --format=zip --output=$sourceZip $SourceCommit .gitattributes src root/f4 "$packetBase/f4_h2_native_budget_seed_20261006.json" "$packetBase/f4_h2_compact_data_pc_packet_20261007" "$packetBase/f4_h2_compact_data_pc_audit_20261007.json"
if ($LASTEXITCODE -ne 0) { throw 'Source/data-export fejlede; ingen rateprøve startet.' }
$sourceRoot=Join-Path $snapshot 'code'
Expand-Archive -LiteralPath $sourceZip -DestinationPath $sourceRoot
$inputBook=Join-Path $RepoRoot 'files/input/data/input.xlsx'
if (-not (Test-Path -LiteralPath $inputBook)) { throw "Mangler eksisterende workbook: $inputBook" }
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$output=Join-Path $RepoRoot "$packetBase/f4_h2_compact_rates_pc_$stamp"
$campaign=Join-Path $RepoRoot "$packetBase/f4_h2_remaining_budget.json"
$launchSeconds=$launchWatch.Elapsed.TotalSeconds.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
python -B (Join-Path $sourceRoot 'root/f4/measure_h2_compact_rates.py') --input-workbook $inputBook --output $output --campaign $campaign --source-commit $SourceCommit --max-seconds $Seconds --launcher-seconds $launchSeconds
$rateExit=$LASTEXITCODE
Write-Host "Send review-ZIP (ogsaa ved stop/fejl): $($output)_review.zip"
if ($rateExit -ne 0) { throw "F4-H2 rateproeve stoppede. Bevar $output og ledgeren; ingen automatisk genkoersel." }
