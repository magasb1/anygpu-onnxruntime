# AnyGpu.OnnxRuntime

Like `AnyCPU`, for the GPU: one .NET 8 code base that runs ONNX Runtime on the best execution provider the machine has,
on **win-x64, linux-x64 and linux-arm64**, falling back to CPU. Nothing is built from source.

- `AnyGpu.OnnxRuntime`: a session factory that tries TensorRT → CUDA → DirectML → CPU and logs why each fallback happened.
- `AnyGpu.OnnxRuntime.Native.linux-arm64`: NVIDIA's prebuilt Jetson ONNX Runtime (CUDA, TensorRT) in NuGet's RID layout,
  the one platform Microsoft's GPU packages skip.

## Supported hardware

| Hardware | Providers | Native package |
|---|---|---|
| NVIDIA GPU, Linux x64 | TensorRT, CUDA | `Microsoft.ML.OnnxRuntime.Gpu.Linux` |
| NVIDIA Jetson, JetPack 6 | TensorRT, CUDA | `AnyGpu.OnnxRuntime.Native.linux-arm64` (this repo) |
| NVIDIA GPU, Windows x64 | TensorRT, CUDA | provider DLLs of `Gpu.Windows` beside DirectML (needs CUDA 12 + cuDNN 9) |
| Any DirectX 12 GPU, Windows 10 1903+ | DirectML | `Microsoft.ML.OnnxRuntime.DirectML` |
| Any x64 or arm64 CPU | CPU | always present |

Not covered (CPU only): DGX Spark / Grace (CUDA 13, SBSA), AMD ROCm on Linux, JetPack 7 ([docs/jetpack7.md](docs/jetpack7.md)).

## Use it

```xml
<PackageReference Include="AnyGpu.OnnxRuntime" Version="1.24.4" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML"         Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'win-x64'" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Windows"      Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'win-x64'" ExcludeAssets="all" GeneratePathProperty="true" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux"        Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'linux-x64'" />
<PackageReference Include="AnyGpu.OnnxRuntime.Native.linux-arm64"  Version="1.24.0" Condition="'$(RuntimeIdentifier)' == 'linux-arm64'" />
<!-- win-x64: only Gpu.Windows' provider libraries, next to DirectML's onnxruntime.dll -->
<None Condition="'$(RuntimeIdentifier)' == 'win-x64'" Link="%(Filename)%(Extension)" CopyToOutputDirectory="PreserveNewest" ExcludeFromSingleFile="true"
      Include="$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Windows)\runtimes\win-x64\native\onnxruntime_providers_cuda.dll;$(PkgMicrosoft_ML_OnnxRuntime_Gpu_Windows)\runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll" />
```

```csharp
var (session, provider) = OnnxSessionFactory.Create("model.onnx", OnnxSessionFactory.Auto, "/var/cache/myapp", Console.WriteLine);
```

- Pass `"tensorrt"`, `"cuda"`, `"dml"` or `"cpu"` instead of `Auto` to force one; it throws if unavailable.
- `configure: o => { ... }` receives the `SessionOptions` before the provider is appended.
- `ANYGPU_TRT_OPTIONS="key=value;..."` overrides TensorRT options per deployment.
- Publish with a runtime identifier; native files land next to the executable, also with `PublishSingleFile=true`.

Why 1.24.x: NVIDIA's JetPack 6 build is 1.24.0 and DirectML stops at 1.24.4. The managed 1.24.4 binding drives both.
On Windows, DirectML's `onnxruntime.dll` is the core and loads the CUDA/TensorRT provider DLLs of `Gpu.Windows` through its
provider bridge, so one build covers every Windows GPU. Without an NVIDIA driver, TensorRT and CUDA are skipped up front.

## Build and run

```powershell
.\get-deps.ps1     # models, test video, Jetson tarball; SHA256-pinned
.\build.ps1        # pack -> artifacts/packages, publish 3 RIDs -> artifacts/publish/<rid>, docker images
.\artifacts\publish\win-x64\AnyGpuOnnx.Example.exe [--ep cpu]
.\artifacts\publish\win-x64\Yolov4.Example.exe
.\artifacts\publish\win-x64\RfDetr.Example.exe --image photo.jpg
.\artifacts\publish\win-x64\LiveDetection.Example.exe --source rtsp://cam1/stream --source rtsp://cam2/stream
docker run --rm anygpu-onnx:cpu
docker run --rm --gpus all anygpu-onnx:cuda
docker run --rm --runtime nvidia -v trt_cache:/trt_cache anygpu-onnx:jetson
```

Common options: `--ep auto|tensorrt|cuda|dml|cpu`, `--trt-cache DIR`, `--model FILE`. Exit code 2: provider unavailable.
Images run `AnyGpuOnnx.Example`; use `--entrypoint /app/<Example>` for the others. Build the Jetson image on the Jetson:
`docker build -f Dockerfile.jetson -t anygpu-onnx:jetson linux-arm64`.

| Example | Shows |
|---|---|
| `AnyGpuOnnx.Example` | one factory call, timed squeezenet |
| `Yolov4.Example` | YOLOv4: letterbox, anchor decoding, NMS |
| `RfDetr.Example` | RF-DETR nano: ImageNet normalisation, no NMS |
| `LiveDetection.Example` | YOLOv4, RF-DETR or Ultralytics YOLO11/YOLO26 on live video feeds, batched |

## Live detection

```
ffmpeg per feed  ->  frame queue       ->  batch + preprocess + Run  ->  decode + sink
decode, resize       keeps the newest      own thread, just in time      NMS, print, stats
```

- **Live, best effort:** video never waits for detection; when inference is slower, the oldest frames are dropped (`dropped=`).
- **Batching across feeds:** `--batch` defaults to the number of `--source`s; waits at most `--max-wait` ms (default 5).
- **Static shapes:** free dimensions are pinned before the session is built; 1.5-3x faster on DirectML than dynamic.
- **No allocation per frame:** two pinned slots bound to ONNX Runtime; decoding one overlaps `Run` on the other.
- **Vectorised preprocessing:** ffmpeg emits the model's layout (planar for NCHW); bytes become floats with SIMD.
- **CPU provider without spin-waiting:** 1.7-1.9x faster on a hybrid Core Ultra 7 265U.

Needs `ffmpeg`/`ffprobe` on PATH (not in the images). Rejected after measuring: int8 and fp16 models, parallel DirectML
sessions, DirectML session tweaks, CPU memory arena off (−65-130 MB, 5-15% slower; set it via `configure` if memory matters more).

Core Ultra 7 265U with Intel iGPU, bundled 12 fps video (647 frames), 1 feed, measured 2026-10-27. Timings vary up to
2-3x with the laptop's thermal state (CPU and iGPU share one power budget):

| Model | DirectML fps / prep / inference / p50 latency | CPU fps / prep / inference / p50 latency |
|---|---|---|
| YOLO26n 640 | 12.0 / 0.3 / 13 / 13 ms | 12.0 / 0.3 / 43 / 42 ms |
| YOLO11n 640 | 12.0 / 0.3 / 13 / 13 ms | 12.0 / 0.3 / 45 / 45 ms |
| RF-DETR nano 384 | 12.0 / 0.2 / 40 / 40 ms | 5.5 / 0.3 / 182 / 223 ms |
| YOLOv4 416 | 10.6 / 0.4 / 94 / 133 ms | 3.2 / 0.4 / 316 / 357 ms |

YOLO11/YOLO26/VisDrone models are [EdgeFirst](https://huggingface.co/EdgeFirst) exports of Ultralytics weights,
**AGPL-3.0**, downloaded for benchmarking only and not bundled. Video: [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), CC BY 4.0.

## Jetson

Orin Nano Super, JetPack 6.2, cached engines:

| Model | TensorRT | TensorRT + `jetson_clocks` | CUDA | CPU |
|---|---|---|---|---|
| squeezenet 1.1 | 1.84 ms | 0.73 ms | 2.62 ms | 8.6 ms |
| YOLOv4 416 | 17.2 ms | 11.8 ms | 68.3 ms | |
| RF-DETR nano | 23.7 ms | 7.7 ms | 28.3 ms | |

- **Run `sudo jetson_clocks`**: the GPU idles at 306 MHz and does not ramp for sporadic inferences (up to 3x faster).
- Persist the TensorRT cache: engine builds take 1-10 min, cached starts 4 s.
- The image installs TensorRT, the ONNX parser and cuDNN from NVIDIA's apt repo; `--runtime nvidia` mounts only the driver.
- New NVIDIA tarball: update URL and SHA256 in `get-deps.ps1` and `JetsonOnnxRuntimeVersion` in `Directory.Build.props`.

## Not verified

- CUDA/TensorRT on x64 Linux and Windows: no NVIDIA x64 machine available. Library loading and CPU fallback are verified.
- Windows arm64.

## License

MIT. ONNX Runtime binaries keep their own licenses (MIT for Microsoft's, NVIDIA's terms for the Jetson build).
