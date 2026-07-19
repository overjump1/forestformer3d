# Run the ForestFormer3D container fully OFFLINE on Windows (Docker Desktop
# with the WSL2 backend; GPU support requires a WSL2-enabled NVIDIA driver).
#
# Usage:
#   .\scripts\run_offline.ps1                # interactive shell in the container
#   .\scripts\run_offline.ps1 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/clean_forestformer/epoch_3000_fix.pth
#
# Env overrides: IMAGE (default forestformer3d:offline), SHM_SIZE (64g),
#                GPUS (all), NETWORK (none).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Image   = if ($env:IMAGE)    { $env:IMAGE }    else { "forestformer3d:offline" }
$ShmSize = if ($env:SHM_SIZE) { $env:SHM_SIZE } else { "64g" }
$Gpus    = if ($env:GPUS)     { $env:GPUS }     else { "all" }
$Network = if ($env:NETWORK)  { $env:NETWORK }  else { "none" }

New-Item -ItemType Directory -Force -Path "data\ForAINetV2", "work_dirs" | Out-Null

docker run --rm -it `
    --gpus $Gpus `
    --shm-size=$ShmSize `
    --network $Network `
    -v "${PWD}\data\ForAINetV2:/workspace/data/ForAINetV2" `
    -v "${PWD}\work_dirs:/workspace/work_dirs" `
    $Image @args
exit $LASTEXITCODE
