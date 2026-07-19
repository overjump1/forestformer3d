# Download the ForAINetV2 dataset (and pretrained model) from Zenodo on
# Windows. ONLINE PHASE - requires internet and a local Python 3 install.
#
# Files land in .\zenodo_downloads (md5-verified, safe to re-run), archives
# are extracted, and the expected layout is assembled:
#   data\ForAINetV2\{train_val_data,test_data}\
#   work_dirs\clean_forestformer\epoch_3000_fix.pth
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Record = if ($env:ZENODO_RECORD) { $env:ZENODO_RECORD } else { "16742708" }
$Dl = "zenodo_downloads"

Write-Host "== Files in Zenodo record ${Record}:"
python scripts/zenodo_fetch.py --record $Record --list
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
python scripts/zenodo_fetch.py --record $Record --out $Dl --extract
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path "data\ForAINetV2", "work_dirs\clean_forestformer" | Out-Null

# Place extracted dataset folders where the pipeline expects them.
foreach ($name in @("train_val_data", "test_data", "meta_data")) {
    $dest = "data\ForAINetV2\$name"
    if (-not (Test-Path $dest)) {
        $src = Get-ChildItem -Path "$Dl\extracted" -Recurse -Directory -Filter $name -ErrorAction SilentlyContinue |
               Select-Object -First 1
        if ($src) {
            Write-Host "Installing $name\ -> $dest"
            Copy-Item -Recurse $src.FullName $dest
        }
    }
}

# Place the pretrained checkpoint.
$ckptDest = "work_dirs\clean_forestformer\epoch_3000_fix.pth"
if (-not (Test-Path $ckptDest)) {
    $ckpt = Get-ChildItem -Path $Dl -Recurse -File -Filter "epoch_3000_fix.pth" -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if (-not $ckpt) {
        $ckpt = Get-ChildItem -Path $Dl -Recurse -File -Filter "*.pth" -ErrorAction SilentlyContinue |
                Select-Object -First 1
    }
    if ($ckpt) {
        Write-Host "Installing checkpoint -> $ckptDest"
        Copy-Item $ckpt.FullName $ckptDest
    }
}

Write-Host ""
Write-Host "== Resulting layout:"
Get-ChildItem "data\ForAINetV2" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
Get-ChildItem "work_dirs\clean_forestformer" -ErrorAction SilentlyContinue | Format-Table Name, Length

if (-not (Test-Path "data\ForAINetV2\train_val_data") -or -not (Test-Path "data\ForAINetV2\test_data")) {
    Write-Host ""
    Write-Host "NOTE: could not auto-detect train_val_data\ and/or test_data\ in the"
    Write-Host "downloaded archives. Inspect $Dl\extracted\ and arrange the files as:"
    Write-Host "  data\ForAINetV2\train_val_data\*.ply"
    Write-Host "  data\ForAINetV2\test_data\*.ply"
}
