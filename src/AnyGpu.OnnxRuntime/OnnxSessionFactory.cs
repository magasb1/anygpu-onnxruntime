using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace AnyGpu.OnnxRuntime;

/// <summary>
/// Creates an <see cref="InferenceSession"/> on the best execution provider the native package next to the executable offers,
/// best first: TensorRT, CUDA, DirectML (Windows), CPU. Each provider is simply tried; a missing provider library or
/// runtime makes ONNX Runtime throw, which is logged and falls through to the next.
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>Provider name meaning "try <see cref="Order"/> until one works".</summary>
    public const string Auto = "auto";

    /// <summary>Every provider this factory knows, best first.</summary>
    public static IReadOnlyList<string> Order { get; } = ["tensorrt", "cuda", "dml", "cpu"];

    // What Auto actually tries: DirectML exists on Windows only; TensorRT and CUDA come from Gpu.Linux, the Jetson package, and on
    // Windows the provider libraries of Gpu.Windows placed next to the DirectML build.
    static readonly string[] PlatformOrder = Order.Where(ep => ep != "dml" || OperatingSystem.IsWindows()).ToArray();

    /// <summary>Creates a session, trying providers until one succeeds.</summary>
    /// <param name="modelPath">Path to the .onnx model.</param>
    /// <param name="provider"><see cref="Auto"/>, or one of <c>tensorrt|cuda|dml|cpu</c> to force it.</param>
    /// <param name="tensorRtCacheDir">Writable folder for TensorRT engine and timing caches. The first build takes minutes, cached starts seconds.</param>
    /// <param name="log">Receives one line per provider that failed and one for the provider chosen.</param>
    /// <param name="configure">Runs on the fresh <see cref="SessionOptions"/> before the provider is appended: threads, profiling, config entries.</param>
    /// <returns>The session and the name of the provider it runs on.</returns>
    /// <exception cref="ArgumentException"><paramref name="provider"/> is not a known name.</exception>
    /// <exception cref="FileNotFoundException"><paramref name="modelPath"/> does not exist.</exception>
    /// <exception cref="InvalidOperationException">No candidate provider could be initialised.</exception>
    public static (InferenceSession Session, string Provider) Create(string modelPath, string provider = Auto, string? tensorRtCacheDir = null, Action<string>? log = null, Action<SessionOptions>? configure = null)
    {
        if (provider != Auto && !Order.Contains(provider))
            throw new ArgumentException($"unknown provider '{provider}' (auto|{string.Join('|', Order)})", nameof(provider));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model not found", modelPath);

        log ??= _ => { };
        List<string> failures = [];
        foreach (string ep in provider == Auto ? PlatformOrder : [provider])
        {
            // Without the NVIDIA driver TensorRT and CUDA cannot work; skipping them here gives one clear line instead of ONNX
            // Runtime's own load errors for each provider library on every machine without an NVIDIA GPU.
            if (ep is "tensorrt" or "cuda" && !HasNvidiaDriver.Value)
            {
                failures.Add($"{ep}: no NVIDIA driver");
                log($"{ep} unavailable: no NVIDIA driver");
                continue;
            }
            try
            {
                using SessionOptions o = Configure(ep, tensorRtCacheDir, configure);
                var session = new InferenceSession(modelPath, o);
                log($"ONNX Runtime {OrtEnv.Instance().GetVersionString()} using {ep}");
                return (session, ep);
            }
            catch (Exception e)
            {
                // EntryPointNotFound: the native package has no such provider at all (e.g. CUDA asked of the DirectML build).
                string why = e is EntryPointNotFoundException ? "not in this ONNX Runtime build" : FirstLine(e);
                failures.Add($"{ep}: {why}");
                log($"{ep} unavailable: {why}");
            }
        }
        throw new InvalidOperationException("No ONNX execution provider available. " + string.Join("; ", failures));
    }

    static SessionOptions Configure(string ep, string? trtCacheDir, Action<SessionOptions>? configure)
    {
        SessionOptions o = new()
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
        };
        configure?.Invoke(o);
        switch (ep)
        {
            case "tensorrt":
                string cache = Path.GetFullPath(trtCacheDir ?? "trt_cache");
                Directory.CreateDirectory(cache);
                Dictionary<string, string> trtOptions = new()
                {
                    ["device_id"] = "0",
                    ["trt_fp16_enable"] = "1",
                    ["trt_engine_cache_enable"] = "1",
                    ["trt_engine_cache_path"] = cache,
                    ["trt_timing_cache_enable"] = "1",
                    ["trt_timing_cache_path"] = cache,
                };
                // Per-deployment knob, no rebuild: ANYGPU_TRT_OPTIONS="trt_fp16_enable=0;trt_builder_optimization_level=5"
                foreach (string kv in (Environment.GetEnvironmentVariable("ANYGPU_TRT_OPTIONS") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                    if (kv.Split('=', 2) is [string k, string v]) trtOptions[k] = v;
                using (OrtTensorRTProviderOptions trt = new())
                {
                    trt.UpdateOptions(trtOptions);
                    o.AppendExecutionProvider_Tensorrt(trt);
                }
                AppendCuda(o); // nodes TensorRT cannot take stay on the GPU instead of falling to CPU
                break;

            case "cuda":
                AppendCuda(o); // throws when the provider library or the CUDA runtime it needs is missing
                break;

            case "dml":
                o.AppendExecutionProvider_DML(0); // throws unless the native package is Microsoft.ML.OnnxRuntime.DirectML
                break;

            case "cpu":
                o.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                o.IntraOpNumThreads = Environment.ProcessorCount; // honours container CPU limits, ORT's own default does not
                // Idle threads sleep instead of spin-waiting between ops. Spinning starves whatever else runs on the machine (decoders,
                // post-processing) and, on hybrid CPUs, the slow cores; measured 1.7-1.9x faster on a Core Ultra 7 265U without it.
                o.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                break;
        }
        return o;
    }

    // Through the provider options (V2) API rather than the exported OrtSessionOptionsAppendExecutionProvider_CUDA, which only
    // CUDA builds of onnxruntime export: this way the DirectML build on win-x64 loads onnxruntime_providers_cuda.dll through its
    // provider bridge, the same way it loads TensorRT, so one Windows build tries TensorRT, CUDA, DirectML and CPU.
    static void AppendCuda(SessionOptions o)
    {
        using OrtCUDAProviderOptions cuda = new();
        cuda.UpdateOptions(new Dictionary<string, string> { ["device_id"] = "0" });
        o.AppendExecutionProvider_CUDA(cuda);
    }

    // The user-mode driver every CUDA program loads; present exactly when an NVIDIA driver is installed (or mounted into the container).
    static readonly Lazy<bool> HasNvidiaDriver = new(() =>
    {
        if (!NativeLibrary.TryLoad(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1", out IntPtr handle)) return false;
        NativeLibrary.Free(handle);
        return true;
    });

    static string FirstLine(Exception e) => e.Message.Split('\n')[0].Trim();
}
