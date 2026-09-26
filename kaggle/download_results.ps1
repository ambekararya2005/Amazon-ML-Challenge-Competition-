# Download a kernel's small results (results/, RUN_INFO.json, kernel log) into kaggle\runs\<Name>.
# Sets PYTHONUTF8=1 (the kaggle CLI otherwise crashes writing the log on Windows) and reads
# KAGGLE_API_TOKEN from the user environment for this process only. Never prints the token.
#   .\kaggle\download_results.ps1 -Kernel aryaambekar/amlc2026-pipeline -Name 20260926-v1
#   .\kaggle\download_results.ps1 -Kernel aryaambekar/amlc2026-block-test-france -Name v2-france -All
param(
    [Parameter(Mandatory = $true)][string]$Kernel,
    [Parameter(Mandatory = $true)][string]$Name,
    [switch]$All
)
$env:PYTHONUTF8 = '1'
if (-not $env:KAGGLE_API_TOKEN) { $env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User') }
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "kaggle\runs\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
$kaggle = Join-Path $root ".venv\Scripts\kaggle.exe"
if ($All) {
    & $kaggle kernels output $Kernel -p $out -o
} else {
    & $kaggle kernels output $Kernel -p $out -o --file-pattern '^(results/|RUN_INFO\.json)'
}
exit $LASTEXITCODE
