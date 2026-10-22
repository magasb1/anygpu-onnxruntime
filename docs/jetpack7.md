# JetPack 7 (CUDA 13) support: notes

Status 2026-10-22: not supported yet. The design accommodates it; what is missing is a prebuilt ONNX Runtime GPU
library for the platform, so it would be the first component in this repo built from source.

## What JetPack 7 is

- JetPack 7.0 was Jetson Thor only. JetPack 7.2 (Jetson Linux 39.2) added the Orin family: Ubuntu 24.04, CUDA 13.2,
  TensorRT 10.16, cuDNN 9.20. An Orin on JetPack 6 needs a reflash to get there.
- CUDA 13 on Jetson is the unified arm64 CUDA platform shared with Grace and DGX Spark. One arm64 CUDA 13 build of
  ONNX Runtime with the right GPU architectures (sm_87 Orin, Thor, sm_90 Grace, Blackwell for Spark) could serve all of
  them from a single package, and would remove the "not covered" line in the README for DGX Spark and Grace.
- `AnyGpu.OnnxRuntime.Native.linux-arm64` (JetPack 6, CUDA 12.6) keeps working on JetPack 6 boards. On a JetPack 7
  board its CUDA 12 libraries do not load; the factory logs that and falls back to CPU. Nothing crashes, the GPU is lost.

## Why it is a build, not a download

No prebuilt ONNX Runtime GPU build works on Orin under JetPack 7 (checked 2026-10-22):

| Source | What it is | Result on Orin (sm_87) |
|---|---|---|
| PyPI `onnxruntime-gpu` | x86_64 and Windows only | no aarch64 |
| jetson-ai-lab `sbsa/cu130` 1.24.0 | SBSA build, sm_90 (Grace) | lists CUDA, no kernel image for sm_87 |
| Hugging Face `cuda13-sm121` wheel | Thor / Blackwell | not sm_87 |
| ultralytics/assets aarch64 wheel | unknown architectures | `cudaErrorNoKernelImageForDevice` at runtime |
| Microsoft GitHub releases | `onnxruntime-linux-aarch64` is CPU only | GPU archives exist for x64 only |

`get_available_providers()` is a compile-time list, so a wheel listing CUDA proves nothing; only a real inference run does.

A community build kit, [straga/jetson-jp7-onnxruntime](https://github.com/straga/jetson-jp7-onnxruntime), compiles
ONNX Runtime from source on the Orin Nano itself: about four hours, memory bound, 16 GB swap, `CMAKE_CUDA_ARCHITECTURES=87`.
It produces a Python wheel. Microsoft's standard wheels do not contain `libonnxruntime.so` (the Python module links the
core statically); the Jetson AI Lab wheels do ship it under `onnxruntime/capi/`. Our packages need the C shared library,
so check the wheel, and otherwise build with `--build_shared_lib`. ONNX Runtime 1.27 or
later builds against CUDA 13 without the `--compile_no_warning_as_error` workaround that 1.23 needed.

## What changes in this repo

1. **Build kit**: a Dockerfile plus script, run on the Jetson, that builds ONNX Runtime with `--build_shared_lib
   --use_cuda --use_tensorrt` for the chosen architectures and tars up `libonnxruntime.so`,
   `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so` and `libonnxruntime_providers_tensorrt.so`.
   Publish the tarball as a GitHub release asset of this repo.
2. **get-deps.ps1**: one more item with the asset URL, its SHA256 and `Dir = 'linux-arm64-cuda13'`.
3. **Second native package** `AnyGpu.OnnxRuntime.Native.linux-arm64-cuda13`, packed from `deps/linux-arm64-cuda13`
   into `runtimes/linux-arm64/native`. Both variants share the linux-arm64 RID, so the consumer picks one at publish time
   with a property (for example `-p:LinuxArm64Cuda=13`).
4. **Dockerfile.jetson** variant on the CUDA 13 L4T base image with the r39 apt repo for TensorRT and cuDNN.
5. **Factory**: no change. The managed binding 1.24.4 drives any newer native build, since a newer library serves
   older C API versions. The `ANYGPU_TRT_OPTIONS` knob and the engine cache work the same.

Roughly a hundred lines of script and packaging plus hours of compute. Do it when a JetPack 7 board is available to
prove it on; until then nothing could be verified.

## Sources

- [JetPack 7.2.1 downloads and notes](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1)
- [JetPack 7.2 downloads and notes](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)
- [JetPack 7.2 components on Orin (RidgeRun)](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_7.2/Getting_Started/Components)
- [JetPack 7.0 components on Thor (RidgeRun)](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_AGX_Thor/JetPack_7.0/Getting_Started/Components)
- [jetson-jp7-onnxruntime build kit](https://github.com/straga/jetson-jp7-onnxruntime)
- [NVIDIA forum: compiling onnxruntime-gpu on Thor](https://forums.developer.nvidia.com/t/method-to-compile-onnxruntime-gpu-on-thor/347563)
