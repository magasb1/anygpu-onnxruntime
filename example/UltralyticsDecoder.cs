// Ultralytics YOLO output decoding (YOLO11, YOLO26, and YOLOv8/v9-style heads), used by LiveDetection.Example (linked with a
// Compile Include). Two output layouts, told apart by shape:
//   [N, 4+C, A]  4 box values (cx, cy, w, h) + C class scores per anchor (C = 80 for COCO, A = 8400 at 640): needs NMS (IoU 0.7)
//   [N, 300, 6]  YOLO26 end2end: x1, y1, x2, y2, score, class, already final: no NMS
// `names` are the model's classes (COCO, or e.g. VisDrone's 10). `boxScaleX/Y` turn box values into input pixels: 1 for
// Ultralytics' own exports, the input width and height for exports that emit boxes normalised to 0..1 (EdgeFirst's). Boxes are
// then mapped from the letterboxed input back to the original image and clipped.
using System.Buffers;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;
using Microsoft.ML.OnnxRuntime.Tensors;

static class UltralyticsDecoder
{
    const float ScoreThreshold = 0.25f, IouThreshold = 0.7f; // Ultralytics' predict defaults

    public static List<Detection> Decode(Tensor<float>[] outputs, int index, string[] names, float boxScaleX, float boxScaleY,
                                         float scale, float padX, float padY, int orgW, int orgH)
    {
        DenseTensor<float> output = (DenseTensor<float>)outputs[0];
        int rows = output.Dimensions[1], cols = output.Dimensions[2];
        ReadOnlySpan<float> v = output.Buffer.Span.Slice(index * rows * cols, rows * cols);

        Detection Box(int cls, float score, float x1, float y1, float x2, float y2) => new(names[cls], score,
            Math.Clamp((x1 * boxScaleX - padX) / scale, 0, orgW - 1), Math.Clamp((y1 * boxScaleY - padY) / scale, 0, orgH - 1),
            Math.Clamp((x2 * boxScaleX - padX) / scale, 0, orgW - 1), Math.Clamp((y2 * boxScaleY - padY) / scale, 0, orgH - 1));

        List<Detection> found = [];
        if (cols == 6)
        {
            for (int q = 0; q < rows; q++)
            {
                ReadOnlySpan<float> d = v.Slice(q * 6, 6);
                if (d[4] >= ScoreThreshold) found.Add(Box((int)d[5], d[4], d[0], d[1], d[2], d[3]));
            }
            return found;
        }

        // Class-major: each class row is contiguous, so the best score per anchor is a vectorised running max over 80 rows of
        // A floats; only the few anchors above the threshold then look up which class it was.
        int anchors = cols;
        float[] rented = ArrayPool<float>.Shared.Rent(anchors);
        Span<float> best = rented.AsSpan(0, anchors);
        try
        {
            v.Slice(4 * anchors, anchors).CopyTo(best);
            for (int c = 1; c < rows - 4; c++)
                TensorPrimitives.Max(best, v.Slice((4 + c) * anchors, anchors), best);
            for (int a = 0; a < anchors; a++)
            {
                if (!(best[a] >= ScoreThreshold)) continue; // also skips NaN, which the lookup below would never find
                int cls = 0;
                while (v[(4 + cls) * anchors + a] != best[a]) cls++;
                float cx = v[a], cy = v[anchors + a], w = v[2 * anchors + a], h = v[3 * anchors + a];
                found.Add(Box(cls, best[a], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
        return Nms.PerClass(found, IouThreshold);
    }
}
