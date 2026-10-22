# AnyGpu.OnnxRuntime

One call that creates an ONNX Runtime `InferenceSession` on the best execution provider the native package offers and
falls back when it is not available, logging why:

| Native package | Tries, in order |
|---|---|
| `AnyGpu.OnnxRuntime.Native.linux-arm64` (NVIDIA Jetson, JetPack 6) | TensorRT (FP16, cached engine) → CUDA → CPU |
| `Microsoft.ML.OnnxRuntime.Gpu.Linux` (linux-x64) | TensorRT → CUDA → CPU |
| `Microsoft.ML.OnnxRuntime.DirectML` (win-x64, any GPU) | DirectML → CPU |
| `Microsoft.ML.OnnxRuntime.DirectML` + the TensorRT and CUDA provider libraries of `Gpu.Windows` (win-x64) | TensorRT → CUDA → DirectML → CPU |

TensorRT and CUDA are skipped, with one log line, when no NVIDIA driver is installed.

```csharp
using AnyGpu.OnnxRuntime;

var (session, provider) = OnnxSessionFactory.Create(
    modelPath: "model.onnx",
    provider: OnnxSessionFactory.Auto,     // or "tensorrt" | "cuda" | "dml" | "cpu" to force one
    tensorRtCacheDir: "/var/cache/myapp",  // writable; a TensorRT engine build takes minutes, cached starts seconds
    log: Console.WriteLine,                // one line per provider that failed, one for the winner
    configure: o => o.EnableProfiling = true); // optional: runs on the SessionOptions before the provider is appended
```

TensorRT provider options can be overridden per deployment without a rebuild:
`ANYGPU_TRT_OPTIONS="trt_fp16_enable=0;trt_builder_optimization_level=5"`.

This package holds only managed code. Add one native package per runtime identifier:

```xml
<PackageReference Include="AnyGpu.OnnxRuntime" Version="1.24.4" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML"         Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'win-x64'" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux"        Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'linux-x64'" />
<PackageReference Include="AnyGpu.OnnxRuntime.Native.linux-arm64"  Version="1.24.0" Condition="'$(RuntimeIdentifier)' == 'linux-arm64'" />
```

Publish with a runtime identifier (`dotnet publish -r linux-arm64`). The native files land flat next to the executable,
also with `PublishSingleFile=true`.
