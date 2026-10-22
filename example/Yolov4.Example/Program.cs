// YOLOv4 (onnx/models, 416x416 NHWC, three detection heads) on the best available provider.
// Letterbox -> run -> decode with anchors -> per-class NMS (../Yolov4Decoder.cs).
// Yolov4.Example [--ep auto|tensorrt|cuda|dml|cpu] [--image FILE] [--trt-cache DIR]
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AnyGpu.OnnxRuntime;
using SkiaSharp;

string provider = OnnxSessionFactory.Auto;
string trtCache = Path.Combine(AppContext.BaseDirectory, "trt_cache");
string imagePath = Path.Combine(AppContext.BaseDirectory, "dog.jpg");
string modelPath = Path.Combine(AppContext.BaseDirectory, "yolov4.onnx");
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ep": provider = args[++i].ToLowerInvariant(); break;
        case "--image": imagePath = args[++i]; break;
        case "--trt-cache": trtCache = args[++i]; break;
        case "--model": modelPath = args[++i]; break;
        default:
            Console.WriteLine("usage: Yolov4.Example [--ep auto|tensorrt|cuda|dml|cpu] [--image FILE] [--trt-cache DIR] [--model FILE]");
            return 0;
    }
}

InferenceSession session;
string used;
try
{
    (session, used) = OnnxSessionFactory.Create(modelPath, provider, trtCache, Console.WriteLine);
}
catch (Exception e) // unknown provider, missing model, or nothing available
{
    Console.Error.WriteLine(e.Message);
    return 2;
}

using (session)
{
    // Input: [1, 416, 416, 3] RGB in 0..1. Letterbox with grey padding, as the reference implementation does.
    (string inputName, NodeMetadata meta) = session.InputMetadata.First();
    int size = meta.Dimensions[1];
    using SKBitmap original = SKBitmap.Decode(imagePath) ?? throw new FileNotFoundException(imagePath);
    int orgW = original.Width, orgH = original.Height;
    float scale = Math.Min((float)size / orgW, (float)size / orgH);
    int newW = (int)(scale * orgW), newH = (int)(scale * orgH);
    int padX = (size - newW) / 2, padY = (size - newH) / 2;
    using SKBitmap resized = original.Resize(new SKImageInfo(newW, newH, SKColorType.Rgb888x, SKAlphaType.Opaque), new SKSamplingOptions(SKFilterMode.Linear));

    var input = new DenseTensor<float>([1, size, size, 3]);
    input.Fill(128 / 255f);
    ReadOnlySpan<byte> px = resized.GetPixelSpan(); // R, G, B, x per pixel
    for (int y = 0; y < newH; y++)
        for (int x = 0; x < newW; x++)
        {
            int o = (y * newW + x) * 4;
            input[0, padY + y, padX + x, 0] = px[o] / 255f;
            input[0, padY + y, padX + x, 1] = px[o + 1] / 255f;
            input[0, padY + y, padX + x, 2] = px[o + 2] / 255f;
        }
    var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, input) };

    List<Detection> detections = [];
    double ms = 0;
    for (int i = 0; i < 4; i++) // the first run warms up (on the Jetson TensorRT builds or loads its engine here)
    {
        long t0 = Stopwatch.GetTimestamp();
        using var results = session.Run(inputs);
        ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        detections = Yolov4Decoder.Decode(results.Select(r => r.AsTensor<float>()).ToArray(), 0, scale, padX, padY, orgW, orgH);
    }

    Console.WriteLine($"RESULT provider={used} image={Path.GetFileName(imagePath)} {orgW}x{orgH} inference={ms:F1}ms detections={detections.Count}");
    foreach (Detection d in detections.OrderByDescending(d => d.Score))
        Console.WriteLine($"  {d}");
}
return 0;
