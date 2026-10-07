param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [Parameter(Mandatory=$true)][ValidateSet('prepare','probe','run','final')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [ValidateRange(76,36000)][double]$Seconds=180,
    [switch]$Resume
)
$ErrorActionPreference='Stop'
$launchWatch=[System.Diagnostics.Stopwatch]::StartNew()
$RepoRoot=(Resolve-Path $RepoRoot).Path
$snapshot=Join-Path ([System.IO.Path]::GetTempPath()) ('f4_h2_staged_'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $snapshot | Out-Null
$sourceZip=Join-Path $snapshot 'source_export.zip'
$packetBase='files/output/bm_1/runs/estimation'
git -C $RepoRoot archive --format=zip --output=$sourceZip $SourceCommit .gitattributes src root/f4 root/H2_COMPACT_STAGED_RUN.md "$packetBase/f4_h2_native_budget_seed_20261006.json" "$packetBase/f4_h2_compact_data_pc_packet_20261007" "$packetBase/f4_h2_compact_data_pc_audit_20261007.json" "$packetBase/f4_h2_compact_rates_pc_review_20261007.zip" "$packetBase/f4_h2_compact_rates_pc_audit_20261007.json"
if ($LASTEXITCODE -ne 0) { throw 'Immutable source/receipt-export fejlede.' }
$sourceRoot=Join-Path $snapshot 'code'
Expand-Archive -LiteralPath $sourceZip -DestinationPath $sourceRoot
$campaign=Join-Path $RepoRoot "$packetBase/f4_h2_remaining_budget.json"
$launchSeconds=$launchWatch.Elapsed.TotalSeconds.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture)
$arguments=@('-B',(Join-Path $sourceRoot 'root/f4/run_h2_compact.py'),$Stage,'--output',$RunDirectory,'--campaign',$campaign,'--source-commit',$SourceCommit,'--max-seconds',$Seconds.ToString('R',[System.Globalization.CultureInfo]::InvariantCulture),'--launcher-seconds',$launchSeconds)
if ($Stage -eq 'final') {
    $arguments+=@('--input-workbook',(Join-Path $RepoRoot 'files/input/data/input.xlsx'))
}
if ($Resume) { $arguments+='--resume' }
python @arguments
$runExit=$LASTEXITCODE
Write-Host "Bevar run-mappen og ledgeren. Review-ZIP: $($RunDirectory)_$($Stage)_review.zip"
if ($runExit -ne 0) { throw 'F4-etapen stoppede; ingen automatisk genkoersel. Send review-ZIP.' }
