# Build the ForestFormer3D image on Windows (Docker Desktop).
# ONLINE PHASE - requires internet access.
# The image targets an RTX A5000 (compute capability 8.6) by default; for a
# different GPU pass e.g.:
#   .\scripts\build.ps1 --build-arg CUDA_ARCH_LIST="8.0"
# Extra args are passed straight to `docker build`, e.g.:
#   .\scripts\build.ps1 --build-arg SKIP_CHECKPOINT=1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Image = if ($env:IMAGE) { $env:IMAGE } else { "forestformer3d:offline" }
docker build -t $Image @args .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Built $Image."
Write-Host "To move it to an air-gapped machine:"
Write-Host "  docker save $Image -o forestformer3d-offline.tar"
Write-Host "  # then on the target machine:"
Write-Host "  docker load -i forestformer3d-offline.tar"
