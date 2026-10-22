# AnyGpu.OnnxRuntime.Native.linux-arm64

The linux-arm64 files that `Microsoft.ML.OnnxRuntime.Gpu.Linux` does not ship: ONNX Runtime 1.24.0 core plus the CUDA and
TensorRT execution providers from NVIDIA's Jetson AI Lab build for JetPack 6 (CUDA 12.6, cuDNN 9, TensorRT 10.3), in the
same `runtimes/linux-arm64/native` layout and with the same `Microsoft.ML.OnnxRuntime.Managed` dependency as Microsoft's
RID packages. Reference it next to them, conditioned on the runtime identifier:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux"        Version="1.24.4" Condition="'$(RuntimeIdentifier)' == 'linux-x64'" />
<PackageReference Include="AnyGpu.OnnxRuntime.Native.linux-arm64"  Version="1.24.0" Condition="'$(RuntimeIdentifier)' == 'linux-arm64'" />
```

The GPU providers load on a JetPack 6 host, or in a container that has the CUDA runtime, cuDNN 9 and TensorRT 10 installed.
The CPU provider runs on any linux-arm64. The managed binding 1.24.4 drives this 1.24.0 native build: it only asks the
library for its C API by number.
