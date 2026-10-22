<#
.SYNOPSIS
Packs the two NuGet packages into artifacts/packages, publishes the examples from them into artifacts/publish/<rid>,
and builds the container images from example/Dockerfile.*.

  artifacts/packages        AnyGpu.OnnxRuntime (.nupkg, .snupkg) + AnyGpu.OnnxRuntime.Native.linux-arm64
  artifacts/publish/<rid>   the four examples, self-contained single-file, native files beside them
  anygpu-onnx:cpu        linux/amd64  runtime-deps, GPU providers removed     CPU fallback
  anygpu-onnx:cpu-arm64  linux/arm64  runtime-deps, GPU providers removed     CPU fallback with the Jetson core
  anygpu-onnx:cuda       linux/amd64  nvidia/cuda + cuDNN                     CUDA, run with --gpus all
  anygpu-onnx:jetson     linux/arm64  l4t-cuda + TensorRT                     TensorRT, run with --runtime nvidia; build it on the Jetson (-Image jetson forces it here)

.EXAMPLE
.\build.ps1                                     # everything
.\build.ps1 -Rid win-x64 -Image none            # pack + publish only: TensorRT, CUDA, DirectML, CPU on Windows
#>
[CmdletBinding()]
param(
    [ValidateSet('win-x64', 'linux-x64', 'linux-arm64')]
    [string[]]$Rid = @('win-x64', 'linux-x64', 'linux-arm64'),

    [ValidateSet('cpu', 'cuda', 'jetson', 'none')]
    [string[]]$Image = @('cpu', 'cuda')
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pkgs = Join-Path $root 'artifacts\packages'

if (-not (Test-Path (Join-Path $root 'deps\linux-arm64\libonnxruntime.so')) -or -not (Test-Path (Join-Path $root 'deps\model\rf-detr-nano.onnx')) -or -not (Test-Path (Join-Path $root 'deps\video\person-bicycle-car-detection.mp4')) -or -not (Test-Path (Join-Path $root 'deps\model\rf-detr-nano-dynamic.onnx'))) {
    & (Join-Path $root 'get-deps.ps1')
}

Write-Host '== pack'
foreach ($p in 'AnyGpu.OnnxRuntime', 'AnyGpu.OnnxRuntime.Native') {
    dotnet pack (Join-Path $root "src\$p") -c Release -o $pkgs -p:NuGetAudit=false --nologo -v q
    if ($LASTEXITCODE -ne 0) { throw "pack $p failed" }
}
Get-ChildItem $pkgs | ForEach-Object { Write-Host ("  {0,-55} {1,7:N0} MB" -f $_.Name, ($_.Length / 1MB)) }

# The versions only change with ONNX Runtime, so evict stale copies from the global package cache before restoring.
$cache = if ($env:NUGET_PACKAGES) { $env:NUGET_PACKAGES } else { Join-Path $HOME '.nuget\packages' }
Get-ChildItem $cache -Directory -Filter 'anygpu.onnxruntime*' -ErrorAction Ignore | Remove-Item -Recurse -Force

# All examples publish into the same folder per RID: they share the native files, so the images carry all four.
foreach ($r in $Rid) {
    Remove-Item (Join-Path $root "artifacts\publish\$r") -Recurse -Force -ErrorAction Ignore   # no stale files from an earlier publish
    foreach ($ex in 'AnyGpuOnnx.Example', 'Yolov4.Example', 'RfDetr.Example', 'LiveDetection.Example') {
        Write-Host "== publish $ex $r"
        dotnet publish (Join-Path $root "example\$ex") -c Release -r $r -o (Join-Path $root "artifacts\publish\$r") -p:NuGetAudit=false --nologo -v q
        if ($LASTEXITCODE -ne 0) { throw "publish $ex $r failed" }
    }
}

$images = @(
    @{ Name = 'cpu';    File = 'Dockerfile.cpu';    Platform = 'linux/amd64'; Rid = 'linux-x64';   Tag = 'anygpu-onnx:cpu' }
    @{ Name = 'cpu';    File = 'Dockerfile.cpu';    Platform = 'linux/arm64'; Rid = 'linux-arm64'; Tag = 'anygpu-onnx:cpu-arm64' }
    @{ Name = 'cuda';   File = 'Dockerfile.cuda';   Platform = 'linux/amd64'; Rid = 'linux-x64';   Tag = 'anygpu-onnx:cuda' }
    @{ Name = 'jetson'; File = 'Dockerfile.jetson'; Platform = 'linux/arm64'; Rid = 'linux-arm64'; Tag = 'anygpu-onnx:jetson' }
)
foreach ($img in $images | Where-Object { $_.Name -in $Image -and $_.Rid -in $Rid }) {
    Write-Host "== docker build $($img.Tag) ($($img.Platform))"
    docker build -f (Join-Path $root "example\$($img.File)") --platform $img.Platform -t $img.Tag (Join-Path $root "artifacts\publish\$($img.Rid)")
    if ($LASTEXITCODE -ne 0) { throw "docker build $($img.Tag) failed" }
}
Write-Host 'done'
