// RF-DETR nano (Roboflow, COCO) on the best available provider.
// Stretch-resize + ImageNet normalisation -> run -> sigmoid on the logits, keep queries above the threshold (../RfDetrDecoder.cs).
// RfDetr.Example [--ep auto|tensorrt|cuda|dml|cpu] [--image FILE] [--trt-cache DIR]
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AnyGpu.OnnxRuntime;
using SkiaSharp;

string provider = OnnxSessionFactory.Auto;
string trtCache = Path.Combine(AppContext.BaseDirectory, "trt_cache");
string imagePath = Path.Combine(AppContext.BaseDirectory, "dog.jpg");
string modelPath = Path.Combine(AppContext.BaseDirectory, "rf-detr-nano.onnx");
int size = 384; // used when the export leaves height and width dynamic (the onnx-community exports do); RF-DETR nano was trained at 384
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ep": provider = args[++i].ToLowerInvariant(); break;
        case "--image": imagePath = args[++i]; break;
        case "--trt-cache": trtCache = args[++i]; break;
        case "--model": modelPath = args[++i]; break; // any RF-DETR export: [1,3,H,W] in, boxes [1,Q,4] + logits [1,Q,C] out
        case "--size": size = int.Parse(args[++i]); break;
        default:
            Console.WriteLine("usage: RfDetr.Example [--ep auto|tensorrt|cuda|dml|cpu] [--image FILE] [--trt-cache DIR] [--model FILE] [--size N]");
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
    // Input: [1, 3, H, W] RGB, stretched to the model size (no letterbox), ImageNet mean/std normalised.
    (string inputName, NodeMetadata meta) = session.InputMetadata.First();
    int height = meta.Dimensions[2] > 0 ? meta.Dimensions[2] : size, width = meta.Dimensions[3] > 0 ? meta.Dimensions[3] : size;
    float[] mean = [0.485f, 0.456f, 0.406f], std = [0.229f, 0.224f, 0.225f];
    using SKBitmap original = SKBitmap.Decode(imagePath) ?? throw new FileNotFoundException(imagePath);
    int orgW = original.Width, orgH = original.Height;
    using SKBitmap resized = original.Resize(new SKImageInfo(width, height, SKColorType.Rgb888x, SKAlphaType.Opaque), new SKSamplingOptions(SKFilterMode.Linear));

    var input = new DenseTensor<float>([1, 3, height, width]);
    ReadOnlySpan<byte> px = resized.GetPixelSpan(); // R, G, B, x per pixel
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int o = (y * width + x) * 4;
            input[0, 0, y, x] = (px[o] / 255f - mean[0]) / std[0];
            input[0, 1, y, x] = (px[o + 1] / 255f - mean[1]) / std[1];
            input[0, 2, y, x] = (px[o + 2] / 255f - mean[2]) / std[2];
        }
    var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, input) };

    List<Detection> detections = [];
    double ms = 0;
    for (int i = 0; i < 4; i++) // the first run warms up (on the Jetson TensorRT builds or loads its engine here)
    {
        long t0 = Stopwatch.GetTimestamp();
        using var results = session.Run(inputs);
        ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        detections = RfDetrDecoder.Decode(results.Select(r => r.AsTensor<float>()).ToArray(), 0, orgW, orgH);
    }

    Console.WriteLine($"RESULT provider={used} image={Path.GetFileName(imagePath)} {orgW}x{orgH} inference={ms:F1}ms detections={detections.Count}");
    foreach (Detection d in detections.OrderByDescending(d => d.Score))
        Console.WriteLine($"  {d}");
}
return 0;

