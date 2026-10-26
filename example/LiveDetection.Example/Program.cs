// Live object detection on one or more video feeds, batched, on the best available provider. YOLOv4 by default; with --model,
// RF-DETR (e.g. rf-detr-nano-dynamic.onnx) and Ultralytics YOLO (YOLO11, YOLO26 end2end) too, told apart by input layout and
// outputs and preprocessed and decoded as each expects.
//
//   ffmpeg per feed  -->  frame queue  -->  batch + preprocess + inference  -->  post-processing + sink
//   decode, resize       keeps the newest   own thread, just in time           decode, NMS, print, stats
//   to model size, RGB24  frames, drops older  for the freshest frames
//
// Live, best effort: the video runs at its own pace and never waits for detection. A file is read at its native frame rate
// (ffmpeg -re), like a camera; an rtsp:// URL works the same way. When inference is slower than the video, the queue drops the
// oldest frames (counted) and detection runs on the newest ones, so results lag the picture by about one Run, never more.
// Several --source arguments share one session and one batch, which is where batching pays off: many cameras, one GPU. The
// batch defaults to the number of feeds, one frame per camera per Run, as DeepStream's batch-size convention does; the batch
// is taken as soon as the device is free, waiting at most --max-wait ms for the rest.
//
// Performance: frames are collected only when the device is free, never prepared ahead to sit in a queue: preprocessing takes
// a few ms, a queued batch would age by a whole Run. Two slots (pinned input and output buffers, bound to ONNX Runtime once)
// alternate between inference and the sink, so decoding overlaps the next Run and nothing is allocated per frame.
// LiveDetection.Example [--ep auto|tensorrt|cuda|dml|cpu] [--source FILE|URL]... [--batch N] [--max-wait MS] [--trt-cache DIR] [--model FILE]
using System.Buffers;
using System.ComponentModel;
using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AnyGpu.OnnxRuntime;

string provider = OnnxSessionFactory.Auto;
string trtCache = Path.Combine(AppContext.BaseDirectory, "trt_cache");
string modelPath = Path.Combine(AppContext.BaseDirectory, "yolov4.onnx");
List<string> sources = [];
int maxBatch = 0;
double maxWaitMs = 5;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--ep": provider = args[++i].ToLowerInvariant(); break;
        case "--source": sources.Add(args[++i]); break;
        case "--batch": maxBatch = int.Parse(args[++i]); break;
        case "--max-wait": maxWaitMs = double.Parse(args[++i]); break;
        case "--trt-cache": trtCache = args[++i]; break;
        case "--model": modelPath = args[++i]; break;
        default:
            Console.WriteLine("usage: LiveDetection.Example [--ep auto|tensorrt|cuda|dml|cpu] [--source FILE|URL]... [--batch N] [--max-wait MS] [--trt-cache DIR] [--model FILE]");
            return 0;
    }
}
if (sources.Count == 0) sources.Add(Path.Combine(AppContext.BaseDirectory, "person-bicycle-car-detection.mp4"));

InferenceSession session;
string used, inputName;
bool planar;
int inW, inH;
Detector detector;
string[] names = Coco.Names80;
float boxScaleX = 1, boxScaleY = 1;
try
{
    // Read the input shape without optimising, to pin every free dimension before the real session is built: batch to --batch,
    // height and width to the model size. Static shapes let DirectML compile the graph once and TensorRT build one engine without
    // profiles; on DirectML they ran 1.5-3x faster per frame than the same model left dynamic. Partial batches are padded.
    int[] dims;
    string[] symbolic;
    int outputCount;
    Dictionary<string, string> metadata;
    using (SessionOptions plain = new() { GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL })
    using (InferenceSession probe = new(modelPath, plain))
    {
        (inputName, NodeMetadata meta) = probe.InputMetadata.First();
        (dims, symbolic) = (meta.Dimensions, meta.SymbolicDimensions);
        outputCount = probe.OutputMetadata.Count;
        metadata = probe.ModelMetadata.CustomMetadataMap;
    }
    // [N, H, W, 3] in: YOLOv4. [N, 3, H, W] in: Ultralytics YOLO with one output, RF-DETR with two (boxes, logits).
    planar = dims[1] == 3;
    detector = !planar ? Detector.Yolov4 : outputCount == 1 ? Detector.Ultralytics : Detector.RfDetr;
    (inH, inW) = planar ? (dims[2], dims[3]) : (dims[1], dims[2]);
    // Free height and width: Ultralytics models were trained at 640; RF-DETR's onnx-community exports leave them free, nano was trained at 384.
    int fallback = detector == Detector.Ultralytics ? 640 : 384;
    if (inH <= 0) inH = fallback;
    if (inW <= 0) inW = fallback;
    if (detector == Detector.Ultralytics)
    {
        // Ultralytics and EdgeFirst exports name their classes in the metadata (COCO's 80, VisDrone's 10, a custom set).
        if (metadata.TryGetValue("labels", out string? labels) || metadata.TryGetValue("names", out labels))
            names = ParseLabels(labels) ?? names;
        // EdgeFirst's exports emit boxes normalised to 0..1, Ultralytics' own export in input pixels.
        if (metadata.ContainsKey("edgefirst")) (boxScaleX, boxScaleY) = (inW, inH);
    }
    if (maxBatch == 0) maxBatch = dims[0] > 0 ? dims[0] : sources.Count; // a fixed batch, or one frame per feed
    if (dims[0] > 0 && dims[0] != maxBatch)
        throw new ArgumentException($"{Path.GetFileName(modelPath)} has a fixed batch of {dims[0]}; use --batch {dims[0]} or an export with a dynamic batch");

    (session, used) = OnnxSessionFactory.Create(modelPath, provider, trtCache, Console.WriteLine, o =>
    {
        int[] pinned = planar ? [maxBatch, 3, inH, inW] : [maxBatch, inH, inW, 3];
        for (int d = 0; d < dims.Length; d++)
            if (dims[d] < 0 && symbolic[d] != "") o.AddFreeDimensionOverrideByName(symbolic[d], pinned[d]);
    });
}
catch (Exception e) // unknown provider, missing or unsuitable model, or nothing available
{
    Console.Error.WriteLine(e.Message);
    return 2;
}

using (session)
using (RunOptions runOptions = new())
{
    // ffmpeg delivers RGB24; a per-channel table turns bytes into model values: 0..1 for YOLOv4 (interleaved) and Ultralytics
    // (planes), ImageNet-normalised planes for RF-DETR.
    int frameBytes = inW * inH * 3;
    long[] inputShape = planar ? [maxBatch, 3, inH, inW] : [maxBatch, inH, inW, 3];
    float[][] lut = detector == Detector.RfDetr ? [Table(0.485f, 0.229f), Table(0.456f, 0.224f), Table(0.406f, 0.225f)] : [Table(0, 1), Table(0, 1), Table(0, 1)];
    // Letterbox grey as each model was trained with (YOLOv4 128, Ultralytics 114); RF-DETR is stretched.
    string? padColor = detector switch { Detector.Yolov4 => "0x808080", Detector.Ultralytics => "0x727272", _ => null };
    string[] inputNames = [inputName], outputNames = [.. session.OutputMetadata.Keys];

    // Warm-up before the first live frame (on the Jetson TensorRT builds or loads its engine here). With every input dimension
    // pinned the output shapes are fixed too, so they size the slots' output buffers.
    long[][] outputShapes;
    using (OrtValue warm = OrtValue.CreateTensorValueFromMemory(new float[maxBatch * frameBytes], inputShape))
    using (IDisposableReadOnlyCollection<OrtValue> outs = session.Run(runOptions, inputNames, [warm], outputNames))
        outputShapes = outs.Select(o => o.GetTensorTypeAndShape().Shape).ToArray();
    // Two slots: one running, one being decoded. A third could only be filled ahead of time, with frames that then age.
    Slot[] slots = Enumerable.Range(0, 2).Select(_ => new Slot(maxBatch, frameBytes, inputShape, outputShapes)).ToArray();

    Feed[] feeds;
    try
    {
        feeds = sources.Select((s, i) => Feed.Open(i, s, inW, inH, padColor)).ToArray();
    }
    catch (Exception e) when (e is Win32Exception or InvalidOperationException) // no ffmpeg on PATH, or a source it cannot open
    {
        Console.Error.WriteLine(e is Win32Exception ? $"ffmpeg and ffprobe must be on PATH: {e.Message}" : e.Message);
        return 1;
    }
    foreach (Feed f in feeds)
        Console.WriteLine($"feed {f.Id}: {f.Source} {f.Width}x{f.Height}");
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; foreach (Feed f in feeds) f.Stop(); }; // Ctrl+C: stop the feeds, the pipeline drains

    // Stage 1, one reader per feed: raw frames from ffmpeg's stdout into the shared queue. Pixel buffers are pooled; a frame the
    // queue drops goes straight back to the pool. The queue holds one batch: the newest frames, never a backlog.
    long dropped = 0;
    Channel<Frame> frames = Channel.CreateBounded<Frame>(
        new BoundedChannelOptions(maxBatch) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true },
        f => { Interlocked.Increment(ref dropped); ArrayPool<byte>.Shared.Return(f.Pixels); });
    _ = Task.WhenAll(feeds.Select(feed => Task.Run(async () =>
    {
        Stream pipe = feed.Process.StandardOutput.BaseStream;
        for (long n = 0; ; n++)
        {
            byte[] pixels = ArrayPool<byte>.Shared.Rent(frameBytes);
            try
            {
                await pipe.ReadExactlyAsync(pixels.AsMemory(0, frameBytes));
            }
            catch (Exception e) when (e is EndOfStreamException or IOException) // end of file, or ffmpeg stopped
            {
                ArrayPool<byte>.Shared.Return(pixels);
                return;
            }
            frames.Writer.TryWrite(new Frame(feed, n, pixels, Stopwatch.GetTimestamp())); // never blocks: drops the oldest when full
        }
    }))).ContinueWith(t => frames.Writer.Complete(t.Exception));

    // Stage 2, on a dedicated thread, because Run blocks for the whole batch and would otherwise hold a thread-pool thread that
    // the readers, Parallel.For and the sink need. As soon as a slot is free it takes the newest frames, preprocesses them in
    // parallel straight into the slot's pinned input, and runs; ONNX Runtime writes into the slot's bound output buffers. Slots
    // past Count hold stale frames, whose results are ignored.
    Channel<Slot> free = Channel.CreateUnbounded<Slot>(), done = Channel.CreateUnbounded<Slot>();
    foreach (Slot s in slots) free.Writer.TryWrite(s);
    Thread inference = new(() =>
    {
        try
        {
            while (true)
            {
                Slot slot = free.Reader.ReadAsync().AsTask().GetAwaiter().GetResult();
                if (!Collect(slot)) break;
                long t0 = Stopwatch.GetTimestamp();
                Parallel.For(0, slot.Count, b =>
                {
                    ToModel(slot.Frames[b].Pixels.AsSpan(0, frameBytes), slot.Input.AsSpan(b * frameBytes, frameBytes), planar, lut);
                    ArrayPool<byte>.Shared.Return(slot.Frames[b].Pixels);
                });
                long t1 = Stopwatch.GetTimestamp();
                session.Run(runOptions, inputNames, slot.InputValues, outputNames, slot.OutputValues);
                slot.PrepMs = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds;
                slot.InferenceMs = Stopwatch.GetElapsedTime(t1).TotalMilliseconds;
                done.Writer.TryWrite(slot);
            }
            done.Writer.Complete();
        }
        catch (Exception e)
        {
            done.Writer.Complete(e);
        }
    }) { IsBackground = true, Name = "inference" };
    inference.Start();

    // Fills the slot with the queued frames, waiting at most --max-wait ms for the rest of the batch. False when the feeds have ended.
    bool Collect(Slot slot)
    {
        if (!frames.Reader.WaitToReadAsync().AsTask().GetAwaiter().GetResult()) return false;
        slot.Count = 0;
        long deadline = Stopwatch.GetTimestamp() + (long)(maxWaitMs * Stopwatch.Frequency / 1000);
        while (slot.Count < maxBatch)
        {
            if (frames.Reader.TryRead(out Frame? f)) { slot.Frames[slot.Count++] = f; continue; }
            TimeSpan left = Stopwatch.GetElapsedTime(Stopwatch.GetTimestamp(), deadline);
            if (left <= TimeSpan.Zero) break;
            using CancellationTokenSource timeout = new(left);
            try { if (!frames.Reader.WaitToReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult()) break; }
            catch (OperationCanceledException) { break; }
        }
        return true;
    }

    // Stage 3, here: decode (+ NMS for YOLOv4), frames in parallel, then the sink, through one buffered writer flushed per batch
    // (a console write per line costs more than decoding on Windows). A service would publish the detections (MQTT, Kafka, a
    // database); this one prints them, with throughput and capture-to-result latency every 5 s.
    using StreamWriter stdout = new(Console.OpenStandardOutput(), bufferSize: 1 << 16) { AutoFlush = false };
    Stats total = new(), window = new();
    long started = Stopwatch.GetTimestamp(), windowStart = started;
    List<Detection>[] decoded = new List<Detection>[maxBatch];
    await foreach (Slot slot in done.Reader.ReadAllAsync())
    {
        long t0 = Stopwatch.GetTimestamp();
        Parallel.For(0, slot.Count, i =>
        {
            Feed feed = slot.Frames[i].Feed;
            decoded[i] = detector switch
            {
                Detector.RfDetr => RfDetrDecoder.Decode(slot.Outputs, i, feed.Width, feed.Height),
                Detector.Ultralytics => UltralyticsDecoder.Decode(slot.Outputs, i, names, boxScaleX, boxScaleY, feed.Scale, feed.PadX, feed.PadY, feed.Width, feed.Height),
                _ => Yolov4Decoder.Decode(slot.Outputs, i, feed.Scale, feed.PadX, feed.PadY, feed.Width, feed.Height),
            };
        });
        for (int i = 0; i < slot.Count; i++)
        {
            Frame f = slot.Frames[i];
            double latency = Stopwatch.GetElapsedTime(f.CapturedAt).TotalMilliseconds;
            stdout.Write($"feed {f.Feed.Id} frame {f.Index,5} {latency,5:F0} ms  ");
            stdout.WriteLine(string.Join(", ", decoded[i].OrderByDescending(d => d.Score).Select(d => $"{d.Label} {d.Score:F2} [{d.X1:F0},{d.Y1:F0},{d.X2:F0},{d.Y2:F0}]")));
            total.Latencies.Add(latency);
            window.Latencies.Add(latency);
        }
        double postMs = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        total.Add(slot, postMs);
        window.Add(slot, postMs);
        free.Writer.TryWrite(slot);
        if (Stopwatch.GetElapsedTime(windowStart).TotalSeconds >= 5)
        {
            stdout.WriteLine($"STATS {window.Format(Stopwatch.GetElapsedTime(windowStart).TotalSeconds)} dropped={Interlocked.Read(ref dropped)}");
            window = new();
            windowStart = Stopwatch.GetTimestamp();
        }
        stdout.Flush();
    }

    stdout.WriteLine($"RESULT provider={used} model={Path.GetFileName(modelPath)} feeds={feeds.Length} batch={maxBatch} frames={total.Latencies.Count} " +
                     $"{total.Format(Stopwatch.GetElapsedTime(started).TotalSeconds)} dropped={Interlocked.Read(ref dropped)}");
    stdout.Flush();
    foreach (Slot s in slots) s.Dispose();
    foreach (Feed f in feeds)
    {
        f.Process.WaitForExit();
        if (f.Process.ExitCode != 0 && !f.Stopped)
        {
            Console.Error.WriteLine($"feed {f.Id}: ffmpeg exited with {f.Process.ExitCode}");
            return 1;
        }
    }
}
return 0;

// Class names from model metadata: EdgeFirst's `labels` is a JSON array, Ultralytics' `names` a Python dict {0: 'person', ...}.
static string[]? ParseLabels(string text)
{
    if (text.TrimStart().StartsWith('['))
        return System.Text.Json.JsonSerializer.Deserialize<string[]>(text);
    string[] found = System.Text.RegularExpressions.Regex.Matches(text, @"\d+\s*:\s*['""]([^'""]*)['""]").Select(m => m.Groups[1].Value).ToArray();
    return found.Length > 0 ? found : null;
}

static float[] Table(float mean, float std) => Enumerable.Range(0, 256).Select(v => (v / 255f - mean) / std).ToArray();

// One RGB24 frame into its place in the batch tensor: interleaved (NHWC) or as three planes (NCHW), one table lookup per value.
static void ToModel(ReadOnlySpan<byte> src, Span<float> dst, bool planar, float[][] lut)
{
    float[] r = lut[0], g = lut[1], b = lut[2];
    int pixels = src.Length / 3;
    if (planar)
    {
        Span<float> dr = dst[..pixels], dg = dst.Slice(pixels, pixels), db = dst.Slice(2 * pixels, pixels);
        for (int p = 0, s = 0; p < pixels; p++, s += 3)
        {
            dr[p] = r[src[s]];
            dg[p] = g[src[s + 1]];
            db[p] = b[src[s + 2]];
        }
    }
    else
    {
        for (int s = 0; s < src.Length; s += 3)
        {
            dst[s] = r[src[s]];
            dst[s + 1] = g[src[s + 1]];
            dst[s + 2] = b[src[s + 2]];
        }
    }
}

record Frame(Feed Feed, long Index, byte[] Pixels, long CapturedAt);

// One batch in flight: input and output buffers allocated once, pinned and bound as OrtValues for Run, and wrapped as tensors
// (no copy) for the decoders.
sealed class Slot : IDisposable
{
    public readonly float[] Input;
    public readonly OrtValue[] InputValues, OutputValues;
    public readonly Tensor<float>[] Outputs;
    public readonly Frame[] Frames;
    public int Count;
    public double PrepMs, InferenceMs;

    public Slot(int batch, int frameValues, long[] inputShape, long[][] outputShapes)
    {
        Input = new float[batch * frameValues];
        InputValues = [OrtValue.CreateTensorValueFromMemory(Input, inputShape)];
        float[][] outputs = outputShapes.Select(s => new float[s.Aggregate(1L, (a, d) => a * d)]).ToArray();
        OutputValues = outputs.Select((o, i) => OrtValue.CreateTensorValueFromMemory(o, outputShapes[i])).ToArray();
        Outputs = outputs.Select((o, i) => (Tensor<float>)new DenseTensor<float>(o, outputShapes[i].Select(d => (int)d).ToArray())).ToArray();
        Frames = new Frame[batch];
    }

    public void Dispose()
    {
        foreach (OrtValue v in InputValues.Concat(OutputValues)) v.Dispose();
    }
}

class Stats
{
    public List<double> Latencies = [];
    int batches;
    double prepMs, inferenceMs, postMs;

    public void Add(Slot s, double post) { batches++; prepMs += s.PrepMs; inferenceMs += s.InferenceMs; postMs += post; }

    public string Format(double seconds)
    {
        if (batches == 0) return "no frames";
        List<double> l = Latencies.Order().ToList();
        return $"fps={l.Count / seconds:F1} batch={(double)l.Count / batches:F2} prep={prepMs / batches:F1}ms/batch inference={inferenceMs / batches:F1}ms/batch " +
               $"post={postMs / batches:F1}ms/batch latency p50={l[(l.Count - 1) / 2]:F0}ms p95={l[(int)((l.Count - 1) * 0.95)]:F0}ms";
    }
}

enum Detector { Yolov4, RfDetr, Ultralytics }

// One video source: ffprobe reads its size (needed to map boxes back), then ffmpeg decodes it to the model's input size, RGB24 on stdout:
// letterboxed with `padColor` and the same rounding as Yolov4.Example, or, without one, stretched as RfDetr.Example does.
class Feed
{
    public required int Id, Width, Height, PadX, PadY;
    public required float Scale;
    public required string Source;
    public required Process Process;
    public bool Stopped { get; private set; }

    public static Feed Open(int id, string source, int inW, int inH, string? padColor)
    {
        ProcessStartInfo probe = new("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", source])
            { RedirectStandardOutput = true, RedirectStandardError = true };
        using Process p = Process.Start(probe)!;
        string output = p.StandardOutput.ReadToEnd(), error = p.StandardError.ReadToEnd();
        p.WaitForExit();
        if (p.ExitCode != 0 || output.Split(',', StringSplitOptions.TrimEntries) is not [string w, string h, ..])
            throw new InvalidOperationException($"cannot open {source}: {error.Trim()}");

        int width = int.Parse(w), height = int.Parse(h);
        float scale = Math.Min((float)inW / width, (float)inH / height);
        int newW = (int)(scale * width), newH = (int)(scale * height), padX = (inW - newW) / 2, padY = (inH - newH) / 2;
        string filter = padColor is null
            ? $"scale={inW}:{inH}:flags=bilinear"
            : $"scale={newW}:{newH}:flags=bilinear,pad={inW}:{inH}:{padX}:{padY}:color={padColor}";
        // A file is paced at its frame rate like a camera (-re); a live source is paced by itself and read without input
        // buffering. Slice threading only: frame threading holds back one frame per decoder thread (latency) and starts a thread
        // per core in every feed's process, competing with the CPU provider. -nostdin keeps ffmpeg off the console.
        string[] input = File.Exists(source) ? ["-re"] : ["-fflags", "nobuffer"];
        ProcessStartInfo decode = new("ffmpeg", [
            "-hide_banner", "-loglevel", "error", "-nostdin", "-thread_type", "slice", .. input, "-i", source, "-an",
            "-vf", filter, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]) { RedirectStandardOutput = true };
        return new Feed
        {
            Id = id, Source = source, Width = width, Height = height, Scale = scale, PadX = padX, PadY = padY,
            Process = Process.Start(decode)!,
        };
    }

    public void Stop()
    {
        Stopped = true;
        try { Process.Kill(); } catch (InvalidOperationException) { } // already exited
    }
}
