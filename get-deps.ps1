<#
.SYNOPSIS
Downloads, with pinned SHA256, the test models and NVIDIA's Jetson AI Lab ONNX Runtime build into deps/.

deps/model/squeezenet1.1-7.onnx            classification test model (onnx/models, 5 MB)
deps/model/yolov4.onnx                     YOLOv4 COCO detector (onnx/models, 257 MB)
deps/model/rf-detr-nano.onnx               RF-DETR nano COCO detector (PierreMarieCurie/rf-detr-onnx on Hugging Face, 108 MB)
deps/model/rf-detr-nano-dynamic.onnx       the same network with dynamic batch, height and width, for batching (onnx-community/rfdetr_nano-ONNX, Apache-2.0, 108 MB)
deps/model/dog.jpg                         test image (AlexeyAB/darknet)
deps/video/person-bicycle-car-detection.mp4  test video, 768x432 12 fps 54 s (intel-iot-devkit/sample-videos, Intel, CC BY 4.0, 6 MB)
deps/linux-arm64/libonnxruntime*.so        ORT 1.24.0 core + CUDA + TensorRT providers for JetPack 6 (64 MB) - packed into AnyGpu.OnnxRuntime.Native.linux-arm64

Idempotent: anything already present is skipped. Needs curl.exe and tar.exe (both ship with Windows 10+).
When NVIDIA ships a new tarball: update the Url, the Sha256 and JetsonOnnxRuntimeVersion in Directory.Build.props.
#>
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$deps = Join-Path $PSScriptRoot 'deps'
$dl = Join-Path $deps 'downloads'
New-Item -ItemType Directory -Force $dl | Out-Null

$items = @(
    @{ Url = 'https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx'
       Sha256 = '1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088'; Dir = 'model' }
    @{ Url = 'https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx'
       Sha256 = '1881fe9c506c970d7866cb4bfc33bda791ce46951a3c39c45ace2ff2b9daf369'; Dir = 'model' }
    @{ Url = 'https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx'
       Sha256 = '3fcbba0f68bad4939fdf1c38f432783b95691e2869af3be369780aa5be67abb2'; Dir = 'model' }
    @{ Url = 'https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx'
       Sha256 = '9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee'; Dir = 'model'; Name = 'rf-detr-nano-dynamic.onnx' }
    @{ Url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/dog.jpg'
       Sha256 = '5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866'; Dir = 'model' }
    @{ Url = 'https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4'
       Sha256 = '452b11b7e0efbd019f1d9570d0c790e90416ad4ad29eec6003872d08443140ef'; Dir = 'video' }
    @{ Url = 'https://apt.jetson-ai-lab.io/jp6/cu126/onnxruntime-gpu-1.24.0.tar.gz'
       Sha256 = '4d927f7c953bf5986d1375c25169aa629045da66738894f28c40964dc0304d5d'; Dir = 'linux-arm64'
       Extract = @('libonnxruntime.so.1.24.0', 'libonnxruntime_providers_shared.so', 'libonnxruntime_providers_cuda.so', 'libonnxruntime_providers_tensorrt.so') }
)

foreach ($it in $items) {
    $target = Join-Path $deps $it.Dir
    $name = if ($it.Name) { $it.Name } else { [IO.Path]::GetFileName($it.Url) }   # Name: when the URL's file name is generic
    $file = Join-Path $dl $name
    # libonnxruntime.so.1.24.0 is installed as libonnxruntime.so, the name the managed binding loads.
    $wanted = @(if ($it.Extract) { $it.Extract -replace '\.so\.[\d.]+$', '.so' } else { $name })
    if (-not $Force -and -not ($wanted | Where-Object { -not (Test-Path (Join-Path $target $_)) })) { Write-Host "ok       $($it.Dir): $($wanted -join ', ')"; continue }

    if ($Force -or -not (Test-Path $file)) {
        Write-Host "download $($it.Url)"
        curl.exe -fL --retry 3 --progress-bar -o $file $it.Url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $($it.Url)" }
    }
    $hash = (Get-FileHash $file -Algorithm SHA256).Hash
    if ($hash -ne $it.Sha256) { Remove-Item $file; throw "SHA256 mismatch for $file`n  got      $hash`n  expected $($it.Sha256)" }

    New-Item -ItemType Directory -Force $target | Out-Null
    if ($it.Extract) {
        # bsdtar matches member names as globs; the libs sit under ./lib/, so extract to a temp dir and pick the wanted files out.
        $tmp = Join-Path $dl 'tmp'
        Remove-Item $tmp -Recurse -Force -ErrorAction Ignore
        New-Item -ItemType Directory -Force $tmp | Out-Null
        tar.exe -xf $file -C $tmp @($it.Extract | ForEach-Object { "*$_" })
        if ($LASTEXITCODE -ne 0) { throw "extract failed: $file" }
        Get-ChildItem $tmp -Recurse -File | Where-Object { $_.Name -in $it.Extract } | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $target ($_.Name -replace '\.so\.[\d.]+$', '.so')) -Force
        }
        Remove-Item $tmp -Recurse -Force
    }
    else {
        Copy-Item $file (Join-Path $target $wanted[0]) -Force
    }
    foreach ($w in $wanted) {
        $p = Join-Path $target $w
        if (-not (Test-Path $p)) { throw "expected $p after extraction" }
        Write-Host ("         {0,-40} {1,10:N0} KB" -f $w, ((Get-Item $p).Length / 1KB))
    }
}
Write-Host 'deps ready'
