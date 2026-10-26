# AnyGpu.OnnxRuntime

Like `AnyCPU`, for the GPU: one .NET 8 code base that runs ONNX Runtime on the best execution provider the machine has,
on **win-x64, linux-x64 and linux-arm64**, and falls back to CPU when there is none. Nothing is built from source.

Two pieces. `AnyGpu.OnnxRuntime` is a small session factory that tries TensorRT, CUDA, DirectML and CPU in that order and
logs why each fallback happened. `AnyGpu.OnnxRuntime.Native.linux-arm64` fills the one gap in Microsoft's packages:
`Microsoft.ML.OnnxRuntime.Gpu.Linux` ships linux-x64 only, so this wraps NVIDIA's prebuilt Jetson build in the same
package layout. Four example apps and container images show it working, one of them live detection on video feeds.

## Supported hardware

| Hardware | Provider | How |
|---|---|---|
| NVIDIA GPU, Linux x64 (workstations, DGX A100/H100, cloud) | TensorRT, CUDA | `Microsoft.ML.OnnxRuntime.Gpu.Linux` |
| NVIDIA Jetson, JetPack 6 (Orin) | TensorRT, CUDA | `AnyGpu.OnnxRuntime.Native.linux-arm64` |
| NVIDIA GPU, Windows x64 | TensorRT, CUDA | provider libraries of `Microsoft.ML.OnnxRuntime.Gpu.Windows` beside the DirectML build (CUDA 12 + cuDNN 9 installed) |
| Any DirectX 12 GPU (Intel, AMD, NVIDIA), Windows 10 1903 or later, Windows 11 | DirectML | `Microsoft.ML.OnnxRuntime.DirectML` |
| Any x64 or arm64 CPU (Intel, AMD, Ampere, Graviton, Raspberry Pi) | CPU | fallback, always present |

Not covered: arm64 NVIDIA machines other than Jetson (DGX Spark, Grace: CUDA 13, SBSA), because no prebuilt ONNX
Runtime GPU build exists for them, Microsoft's `linux-aarch64` archive is CPU only; and AMD GPUs on Linux (ROCm), which
Microsoft does not package. Both run on CPU today and would need a from-source build to go further.
JetPack 7 (CUDA 13) on Jetson is in the same position; [docs/jetpack7.md](docs/jetpack7.md) has the notes and the plan.

| Runtime identifier | Native package | Tries, in order |
|---|---|---|
| linux-arm64 (JetPack 6) | `AnyGpu.OnnxRuntime.Native.linux-arm64` (this repo, Jetson AI Lab ORT 1.24.0) | TensorRT → CUDA → CPU |
| linux-x64 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` | TensorRT → CUDA → CPU |
| win-x64 | `Microsoft.ML.OnnxRuntime.DirectML` + the TensorRT and CUDA provider libraries of `Gpu.Windows` | TensorRT → CUDA → DirectML → CPU |

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

`Create` tries the providers in order, logs why each one fell through, and returns the session plus the provider's name.
Pass `"tensorrt"`, `"cuda"`, `"dml"` or `"cpu"` instead of `Auto` to force one; it throws if that one is unavailable.
An optional `configure: o => { ... }` callback receives the `SessionOptions` before the provider is appended.
`ANYGPU_TRT_OPTIONS="key=value;..."` overrides TensorRT provider options per deployment (see Jetson notes).
Publish with a runtime identifier; the native files land flat next to the executable, also with `PublishSingleFile=true`.

The factory is a convenience. The native package works on its own next to `Microsoft.ML.OnnxRuntime.Managed`, like
Microsoft's RID packages do.

## Packages

| Package | Contents | Depends on |
|---|---|---|
| `AnyGpu.OnnxRuntime` 1.24.4 | `OnnxSessionFactory`, managed only, RID-neutral | `Microsoft.ML.OnnxRuntime.Managed` |
| `AnyGpu.OnnxRuntime.Native.linux-arm64` 1.24.0 | `runtimes/linux-arm64/native/libonnxruntime*.so` (core, CUDA, TensorRT), 61 MB | `Microsoft.ML.OnnxRuntime.Managed` |

The native package is versioned as the ONNX Runtime it contains. The managed binding 1.24.4 drives it because it only
asks the native library for its C API by number; the same binding drives Microsoft's 1.24.4 native packages on x64.
The GPU providers in it need a JetPack 6 host or a container with CUDA 12.6, cuDNN 9 and TensorRT 10; the CPU
provider runs on any linux-arm64 (verified under emulation).

On Windows, DirectML and `Gpu.Windows` each ship their own `onnxruntime.dll`, so only one can be the core. One build covers
every Windows GPU: DirectML's core, plus `onnxruntime_providers_cuda.dll` and `onnxruntime_providers_tensorrt.dll` from
`Gpu.Windows` (275 MB), which the DirectML core loads through its provider bridge, as the CUDA core does. The factory appends
CUDA through the provider-options (V2) API for this, since only CUDA builds export the old entry point. Without an NVIDIA
driver (`nvcuda.dll`, `libcuda.so.1` on Linux) the factory skips TensorRT and CUDA with one log line. CUDA on Windows needs
the CUDA 12 and cuDNN 9 runtimes installed, TensorRT also TensorRT 10; otherwise it falls through to DirectML.

The ONNX Runtime line is set by the hardest platform, the Jetson: NVIDIA's JetPack 6 build is 1.24.0, and the managed
binding 1.24.4 drives it. On the Microsoft side DirectML stops at 1.24.4 (core ONNX Runtime has moved on, DirectML has not),
so 1.24.4 is the newest version every RID shares. WebGPU (plugin EP `Microsoft.ML.OnnxRuntime.EP.WebGpu`, which loads into
this same DirectML core) was measured as a non-NVIDIA alternative and left out: 2.5-3x slower than DirectML on an Intel iGPU.

## Repository layout

```
src/AnyGpu.OnnxRuntime/          managed library, OnnxSessionFactory.cs, package README
src/AnyGpu.OnnxRuntime.Native/   packs deps/linux-arm64/*.so as runtimes/linux-arm64/native
example/Directory.Build.props       the package references and publish settings every example shares
example/AnyGpuOnnx.Example/      minimal consumer: one factory call, a timed squeezenet run
example/Yolov4.Example/             YOLOv4 (onnx/models): letterbox, anchor decoding, per-class NMS
example/RfDetr.Example/             RF-DETR nano (Roboflow): ImageNet normalisation, sigmoid + threshold, no NMS
example/LiveDetection.Example/      YOLOv4 or RF-DETR on live video feeds: ffmpeg decode, drop-oldest queue, batching, staged pipeline
example/Coco.cs                     COCO class names shared by the detectors
example/Yolov4Decoder.cs            YOLOv4 anchor decoding + NMS, shared by Yolov4.Example and the live example
example/RfDetrDecoder.cs            RF-DETR decoding, shared by RfDetr.Example and the live example
example/UltralyticsDecoder.cs       Ultralytics YOLO decoding (YOLO11 with NMS, YOLO26 end2end without), for the live example
example/Nms.cs                      per-class NMS, shared by the YOLO decoders
example/Dockerfile.{cpu,cuda,jetson}
get-deps.ps1                        downloads the models, the test image and video and the Jetson tarball into deps/, SHA256-pinned
build.ps1                           pack -> artifacts/packages, publish examples -> artifacts/publish/<rid>, docker build
Directory.Build.props               ONNX Runtime versions and shared package metadata
```

The detectors use SkiaSharp (MIT) for image decoding. Each prints the provider, the inference time and the detections
for `dog.jpg` (or `--image FILE`); the boxes must agree across providers.

## Build and run

```powershell
.\get-deps.ps1                         # models (580 MB), test video and the Jetson tarball (64 MB), checksums verified
.\build.ps1                            # packs, publishes 3 RIDs, builds the cpu, cpu-arm64 and cuda images
.\artifacts\publish\win-x64\AnyGpuOnnx.Example.exe              # DirectML on any Windows GPU
.\artifacts\publish\win-x64\AnyGpuOnnx.Example.exe --ep cpu
.\artifacts\publish\win-x64\Yolov4.Example.exe
.\artifacts\publish\win-x64\RfDetr.Example.exe --image photo.jpg
.\artifacts\publish\win-x64\LiveDetection.Example.exe                    # bundled video, paced like a camera; needs ffmpeg on PATH
.\artifacts\publish\win-x64\LiveDetection.Example.exe --source rtsp://cam1/stream --source rtsp://cam2/stream   # one batch of 2
.\artifacts\publish\win-x64\LiveDetection.Example.exe --model .\artifacts\publish\win-x64\rf-detr-nano-dynamic.onnx
docker run --rm anygpu-onnx:cpu
docker run --rm --entrypoint /app/Yolov4.Example anygpu-onnx:cpu
docker run --rm --platform linux/arm64 anygpu-onnx:cpu-arm64
docker run --rm --gpus all anygpu-onnx:cuda                                  # NVIDIA x64 host
docker run --rm --runtime nvidia -v trt_cache:/trt_cache anygpu-onnx:jetson  # Jetson, JetPack 6
docker run --rm --runtime nvidia -v trt_cache:/trt_cache --entrypoint /app/RfDetr.Example anygpu-onnx:jetson --trt-cache /trt_cache
```

All four examples publish into the same folder per RID and ship in every image; the image entrypoint is
`AnyGpuOnnx.Example`, use `--entrypoint` for the others. Options: `--ep auto|tensorrt|cuda|dml|cpu`, `--trt-cache DIR`;
`--model FILE` to run another export; `--iters N` for the squeezenet example, `--image FILE` for the detectors,
`--source FILE|URL` (repeatable), `--batch N` (default: number of feeds) and `--max-wait MS` for the live example. Exit code 2 means the requested provider is not available.
Build the Jetson image on the Jetson itself: copy `artifacts/publish/linux-arm64` and `example/Dockerfile.jetson` over and run
`docker build -f Dockerfile.jetson -t anygpu-onnx:jetson linux-arm64`.

## Live detection on video

`LiveDetection.Example` runs YOLOv4 (default), RF-DETR (`--model rf-detr-nano-dynamic.onnx`) or Ultralytics YOLO11 / YOLO26
(`--model deps\model\yolo26n.onnx`) on one or more video feeds the way a video analytics service does. The model is recognised
by its input layout and outputs: NHWC is YOLOv4 (letterboxed, grey 128); NCHW with one output is Ultralytics (letterboxed, grey
114; `[N,84,A]` decoded with NMS, YOLO26's end2end `[N,300,6]` without); NCHW with two outputs is RF-DETR (stretched,
ImageNet-normalised).

```
ffmpeg per feed  ->  frame queue        ->  batch + preprocess + Run  ->  post-processing + sink
decode, resize       keeps the newest       own thread, just in time      decode, NMS, print, stats
```

- **Live, best effort.** The video runs at its own pace (`ffmpeg -re` for a file, like a camera) and never waits for
  detection. When inference is slower than the video, the queue (one batch deep) drops the oldest frames, counted in
  `dropped=`, and detection runs on the newest ones, so results trail the picture by about one `Run`, never more.
- **Just in time.** Frames are taken only when the device is free and preprocessed straight into the pinned input, frames
  in parallel. Preparing the next batch during a `Run` would keep the device a few ms busier, but its frames would then age
  by a whole `Run` before inference, so they are never prepared ahead.
- **Batching across feeds.** Several `--source` arguments share one session; `--batch` defaults to the number of feeds (one
  frame per camera per `Run`, DeepStream's convention). The batcher takes what is queued and waits at most `--max-wait` ms
  (default 5) for the rest; a partial batch is padded.
- **Static shapes.** Every free input dimension (batch, and height and width for the onnx-community RF-DETR export) is pinned
  before the session is built (`AddFreeDimensionOverrideByName`). DirectML then compiles once and TensorRT builds a single
  engine without profiles; on the iGPU below that measured 1.5-3x faster per frame than leaving the shapes dynamic. A model
  with a fixed batch of 1, such as `rf-detr-nano.onnx`, runs with `--batch 1` only, which is why the live example ships the
  dynamic export of the same network (onnx-community, identical scores on `dog.jpg`).
- **No allocation per frame.** Two slots of input and output buffers are allocated once, pinned and bound to ONNX Runtime
  as `OrtValue`s; `Run` writes straight into them and the decoders read them in place. One slot runs while the other is
  decoded, so decoding, NMS and output overlap the next `Run`. Pixel buffers are pooled; bytes become model values through a
  per-channel lookup table. Output goes through one buffered writer, flushed per batch.
- **Inference on a dedicated thread**, so a `Run` blocking for hundreds of ms never holds a thread-pool thread the readers
  and the parallel loops need.
- **ffmpeg with slice threading only.** Frame threading holds back one frame per decoder thread (latency the program cannot
  see) and starts a thread per core in every feed's process, competing with the CPU provider.
- **CPU provider without spin-waiting** (in `OnnxSessionFactory`, so for every CPU session): idle ORT threads sleep instead of
  burning the cores the decoders and the sink need. 1.7-1.9x faster per frame on the Core Ultra 7 265U below.
- Per frame it prints capture-to-result latency and the detections (label, score, box in source pixels); every 5 s a `STATS` line, at the end a `RESULT` line
  (fps, mean batch, preprocessing, inference and post-processing per batch, latency p50/p95, frames dropped). Ctrl+C stops
  the feeds and drains the pipeline.

Tried and rejected, measured on the same machine: int8 RF-DETR (10% faster on CPU, 3.5x slower on DirectML), fp16 models
(slower on both), two or three DirectML sessions in parallel (no gain: one session saturates the iGPU), and the
sequential / no-memory-pattern session settings for DirectML (no difference beyond the noise).

Needs `ffmpeg` and `ffprobe` on PATH; the container images do not include them. Any input ffmpeg opens works (file, `rtsp://`, `http://`).
The bundled video is `person-bicycle-car-detection.mp4` from [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos),
by Intel, CC BY 4.0: 768x432, 12 fps, 54 s.

Windows 11, Core Ultra 7 265U (2 performance + 10 efficiency cores) with its Intel iGPU, bundled video played at its own pace
(12 fps, 647 frames per feed), measured 2026-10-26. `fps` is frames that got detections, across all feeds; latency is from
the frame leaving ffmpeg to its detections. The iGPU's `Run` time varies by up to ±50% between identical runs (the CPU and GPU
share one power budget), so read the DirectML rows as rough:

| Provider | Model | Feeds, batch | fps | prep / inference / post per batch | latency p50 / p95 | dropped |
|---|---|---|---|---|---|---|
| DirectML | YOLOv4 | 1, 1 | 4.0 | 2.6 / 243 / 1.3 ms | 302 / 383 ms | 428 of 647 |
| DirectML | RF-DETR nano | 1, 1 | 9.1 | 2.0 / 107 / 0.4 ms | 151 / 203 ms | 157 of 647 |
| DirectML | RF-DETR nano | 2, 2 | 9.8 | 3.0 / 200 / 0.7 ms | 248 / 323 ms | 763 of 1294 |
| CPU | YOLOv4 | 1, 1 | 1.6 | 1.4 / 608 / 3.2 ms | 654 / 770 ms | 558 of 647 |
| CPU | RF-DETR nano | 1, 1 | 2.9 | 1.2 / 347 / 0.7 ms | 381 / 553 ms | 492 of 647 |
| DirectML | YOLO11n | 1, 1 | 12.0 | 1.4 / 13 / 1.0 ms | 15 / 23 ms | 4 of 647 |
| DirectML | YOLO26n | 1, 1 | 12.0 | 1.2 / 13 / 0.1 ms | 14 / 17 ms | 4 of 647 |
| DirectML | YOLO11s | 1, 1 | 11.6 | 3.1 / 51 / 1.7 ms | 48 / 157 ms | 24 of 647 |
| DirectML | YOLO26s | 1, 1 | 11.1 | 4.4 / 76 / 0.2 ms | 88 / 175 ms | 54 of 647 |
| DirectML | VisDrone YOLO26n 640 | 1, 1 | 12.0 | 1.1 / 11 / 0.3 ms | 12 / 14 ms | 4 of 647 |
| DirectML | VisDrone YOLO26n 1344x768 | 1, 1 | 12.0 | 5.0 / 33 / 0.8 ms | 38 / 54 ms | 5 of 647 |
| CPU | YOLO11n | 1, 1 | 9.0 | 1.7 / 107 / 1.5 ms | 150 / 205 ms | 162 of 647 |
| CPU | YOLO26n | 1, 1 | 8.8 | 2.2 / 111 / 0.2 ms | 155 / 211 ms | 176 of 647 |
| CPU | VisDrone YOLO26n 640 | 1, 1 | 9.5 | 1.9 / 104 / 0.5 ms | 146 / 194 ms | 140 of 647 |

The Ultralytics nano models are the only ones here that keep up with the video on this iGPU, at 15 ms from frame to
detections, and they run three to four times as many frames as RF-DETR nano on the CPU. YOLO26's end2end output needs no
NMS, so its post-processing is 0.1-0.2 ms against YOLO11's 1-1.7 ms. Bare `Run` time per frame at batch 1, three interleaved
rounds on DirectML and one on CPU, same machine:

| Model | Input | DirectML | CPU |
|---|---|---|---|
| YOLO11n | 640 | 18-25 ms | 80 ms |
| YOLO26n (end2end) | 640 | 18-24 ms | 125 ms |
| YOLO11s | 640 | 46-70 ms | 324 ms |
| YOLO26s (end2end) | 640 | 47-69 ms | 296 ms |
| RF-DETR nano | 384 | 73-100 ms | 450 ms |
| YOLOv4 | 416 | 104-264 ms | 701 ms |

The YOLO11, YOLO26 and VisDrone models (`get-deps.ps1`) are ONNX exports by [EdgeFirst](https://huggingface.co/EdgeFirst):
YOLO11/YOLO26 are the official Ultralytics COCO weights unchanged, the VisDrone ones EdgeFirst's training on VisDrone2019
(drone views, 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). All are
**AGPL-3.0** (Ultralytics; an Ultralytics Enterprise License is the alternative), and VisDrone's dataset has its own terms.
They are downloaded for benchmarking only: no example bundles them and no container image ships them; pass them with
`--model deps\model\yolo26n.onnx`. The live example reads their class names from the model metadata and accepts
non-square inputs such as the 16:9 1344x768 VisDrone variant, which suits video better than a letterboxed square.
On this street video, filmed from above but not by a drone, the VisDrone models find pedestrians, cars and cyclists; on
`dog.jpg`, a ground-level photo, they are weak, as expected outside their training views.

Against the previous pipeline, alternated with it on the same 30 s clip: the pipeline's own overhead per batch with the device
saturated fell from 16-21 ms to 6-9 ms, and YOLOv4 on DirectML ran 36% more frames (`Run` itself 25% faster with less CPU
churn beside it). On CPU almost all of the gain is the factory's spin-waiting fix, measured on the bare session: YOLOv4 1130-1200 ms ->
580-630 ms per frame, RF-DETR 510-630 ms -> 300-355 ms. What is left is the model itself: preprocessing and post-processing
are 1-3% of a frame.

## Results

squeezenet1.1, batch 1, mean of 100 runs after 3 warm-ups. `top1` is the argmax of a fixed-seed input and matches on every
provider and platform.

| Machine | Command | Provider used | mean ms | top1 |
|---|---|---|---|---|
| Windows 11, Intel iGPU | `AnyGpuOnnx.Example.exe` | DirectML | 2.8 | 111 |
| same | `--ep cpu` | CPU | 8.1 | 111 |
| Docker Desktop, `anygpu-onnx:cpu` | | CPU (GPU providers removed from the image) | 28 | 111 |
| Docker Desktop, `anygpu-onnx:cpu-arm64` (emulated) | | CPU (Jetson core 1.24.0) | 2400 | 111 |
| Docker Desktop, `anygpu-onnx:cuda`, no NVIDIA GPU | | CPU (falls back without `--gpus all`) | 28 | 111 |
| Jetson Orin Nano Super, JetPack 6.2, `anygpu-onnx:jetson`, first start, `jetson_clocks` | | TensorRT FP16, engine build 50 s | 0.73 | 111 |
| same, second start (4 s) | | TensorRT FP16 from cache | 0.72 | 111 |
| same, native | `--ep cuda` | CUDA | 2.6 | 111 |
| same, native | `--ep cpu` | CPU | 8.6 | 111 |

Jetson Orin Nano Super (8 GB, JetPack 6.2), running the published folder natively, measured 2026-10-22. Inference time of
one run after warm-up, cached engines; the detections are identical across providers.

| Model | TensorRT, default clocks | TensorRT, `jetson_clocks` | CUDA, `jetson_clocks` | CPU, `jetson_clocks` |
|---|---|---|---|---|
| squeezenet 1.1 (mean of 200) | 1.84 ms | 0.73 ms | 2.62 ms | 8.6 ms |
| YOLOv4 416 | 17.2 ms | 11.8 ms | 68.3 ms | |
| RF-DETR nano | 23.7 ms | 7.7 ms | 28.3 ms | |

Image sizes (Docker Desktop, 2026-10-22): cpu 1.2 GB, cpu-arm64 492 MB, cuda 7.5 GB, jetson 4.25 GB (built on the Jetson).
The cpu images drop the GPU provider libraries in a throwaway build stage; the rest is the three single-file examples plus 370 MB of models.

## Not verified

- **CUDA and TensorRT on x64**: no NVIDIA x64 machine was available. These are Microsoft's own packages; the fallback to CPU
  is verified. The cuda image has the CUDA runtime and cuDNN but not TensorRT, so it runs CUDA; start from an
  `nvcr.io/nvidia/tensorrt` image to get TensorRT on x64.
- **CUDA and TensorRT on Windows** through the DirectML core: verified that the core loads `onnxruntime_providers_cuda.dll`
  and `onnxruntime_providers_tensorrt.dll` from `Gpu.Windows` 1.24.4 (on a machine without CUDA it stops at the missing
  `cublasLt64_12.dll`), and the two cores carry the same CUDA bridge code. Running a session on an NVIDIA GPU this way is not
  verified: no NVIDIA Windows machine was available. If it fails there, the fallback is DirectML.
- **Windows arm64**: DirectML 1.24.4 supports it; not tried.
- `SessionOptions.SetEpSelectionPolicy(ExecutionProviderDevicePolicy.PREFER_GPU)` is the one-line automatic alternative; the
  explicit loop is used because it can express TensorRT > CUDA > DirectML, reports why each fallback happened, and the
  Jetson build's legacy TensorRT provider is not visible to the device-policy API.

## Jetson notes

- The Jetson AI Lab ORT 1.24.0 build links cuDNN from both the TensorRT and the CUDA provider, and the TensorRT provider
  dlopens `libnvonnxparser.so.10`. JetPack 6's `--runtime nvidia` mounts only the driver and devices, so the image installs
  `libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10 libcudnn9-cuda-12` from NVIDIA's apt repo (about 1.3 GB).
- Outside containers a JetPack 6 host already has everything; run the published folder directly and give it a writable cache dir.
- The TensorRT engine build is keyed to the GPU and takes 1 min (squeezenet), 3 min (RF-DETR nano) or 10 min (YOLOv4)
  on an Orin Nano; persist the cache volume. The timing cache in the same folder speeds up later builds.
- **Run `sudo jetson_clocks`** (or an equivalent nvpmodel/clock policy) on a latency-sensitive box. The GPU idles at 306 MHz
  and DVFS does not ramp for sporadic single inferences: pinned clocks cut squeezenet from 1.84 to 0.73 ms, YOLOv4 from
  17.2 to 11.8 ms and RF-DETR from 23.7 to 7.7 ms on an Orin Nano Super, with no code change.
- `ANYGPU_TRT_OPTIONS="key=value;key=value"` overrides or adds TensorRT provider options per deployment without a
  rebuild, for example `trt_fp16_enable=0` for accuracy-sensitive models or `trt_int8_enable=1` with a calibration table.
- Tried and not adopted: `trt_builder_optimization_level=5` (same latency, twice the build time) and
  `trt_cuda_graph_enable=1` (needs IOBinding with fixed device buffers; with the plain `Run` API the outputs come back unusable).
- When NVIDIA ships a new tarball: update the URL and SHA256 in `get-deps.ps1` and `JetsonOnnxRuntimeVersion` in `Directory.Build.props`.

## License

MIT. The ONNX Runtime binaries keep their own licenses (MIT for Microsoft's, NVIDIA's terms for the Jetson AI Lab build).
