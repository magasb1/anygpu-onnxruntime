// Example consumer: AnyGpuOnnx.Example [--ep auto|tensorrt|cuda|dml|cpu] [--iters N] [--trt-cache DIR]
// Exit 0 = ran, 2 = the requested provider is not available.
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AnyGpu.OnnxRuntime;

string provider = OnnxSessionFactory.Auto;                                   // a service reads these from config
string trtCache = Path.Combine(AppContext.BaseDirectory, "trt_cache");
string model = Path.Combine(AppContext.BaseDirectory, "squeezenet1.1-7.onnx");
int iters = 50;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ep": provider = args[++i].ToLowerInvariant(); break;
        case "--iters": iters = int.Parse(args[++i]); break;
        case "--trt-cache": trtCache = args[++i]; break;
        case "--model": model = args[++i]; break; // any model with one float input: fixed-seed data of its shape, batch 1
        default:
            Console.WriteLine("usage: AnyGpuOnnx.Example [--ep auto|tensorrt|cuda|dml|cpu] [--iters N] [--trt-cache DIR] [--model FILE]");
            return 0;
    }
}

Console.WriteLine($"{RuntimeInformation.OSDescription} {RuntimeInformation.RuntimeIdentifier}, {Environment.ProcessorCount} cores");

InferenceSession session;
string used;
try
{
    (session, used) = OnnxSessionFactory.Create(model, provider, trtCache, Console.WriteLine);
}
catch (Exception e) // unknown provider, missing model, or nothing available
{
    Console.Error.WriteLine(e.Message);
    return 2;
}

using (session)
{
    // Fixed-seed input: the printed top1 index must match on every provider and platform.
    KeyValuePair<string, NodeMetadata> input = session.InputMetadata.First();
    int[] dims = input.Value.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
    float[] data = new float[dims.Aggregate(1, (a, b) => a * b)];
    Random rnd = new(42);
    for (int i = 0; i < data.Length; i++) data[i] = (float)rnd.NextDouble();
    var inputs = new[] { NamedOnnxValue.CreateFromTensor(input.Key, new DenseTensor<float>(data, dims)) };

    int top1 = -1;
    for (int i = 0; i < 3; i++) // startup warm-up; on the Jetson this is where TensorRT builds or loads its engine
    {
        using var r = session.Run(inputs);
        float[] scores = r.First().AsEnumerable<float>().ToArray();
        top1 = Array.IndexOf(scores, scores.Max());
    }

    double[] ms = new double[iters];
    for (int i = 0; i < iters; i++)
    {
        long t0 = Stopwatch.GetTimestamp();
        using var r = session.Run(inputs);
        ms[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    }
    Array.Sort(ms);
    Console.WriteLine($"RESULT provider={used} iters={iters} mean={ms.Average():F2}ms p50={ms[iters / 2]:F2}ms min={ms[0]:F2}ms top1={top1}");
}
return 0;
