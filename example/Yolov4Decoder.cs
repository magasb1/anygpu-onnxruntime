// YOLOv4 (onnx/models) output decoding, shared by Yolov4.Example and LiveDetection.Example (linked with a Compile Include).
// Decoding constants are those of the reference notebook in onnx/models.
using Microsoft.ML.OnnxRuntime.Tensors;

static class Yolov4Decoder
{
    // Heads are [N, S, S, 3, 85] for S = 52, 26, 13: raw xy/wh, then objectness and 80 class probabilities already in 0..1.
    // Decodes image `index` of the batch, maps the boxes from the letterboxed input back to the original image and clips them.
    public static List<Detection> Decode(Tensor<float>[] heads, int index, float scale, float padX, float padY, int orgW, int orgH)
    {
        float[][] anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]];
        int[] strides = [8, 16, 32];
        float[] xyScale = [1.2f, 1.1f, 1.05f];
        const float scoreThreshold = 0.25f, iouThreshold = 0.213f;

        List<Detection> candidates = [];
        Tensor<float>[] byGrid = heads.OrderByDescending(h => h.Dimensions[1]).ToArray();
        for (int h = 0; h < byGrid.Length; h++)
        {
            int grid = byGrid[h].Dimensions[1], perImage = grid * grid * 3 * 85;
            ReadOnlySpan<float> v = ((DenseTensor<float>)byGrid[h]).Buffer.Span.Slice(index * perImage, perImage);
            for (int gy = 0; gy < grid; gy++)
            for (int gx = 0; gx < grid; gx++)
            for (int a = 0; a < 3; a++)
            {
                int o = ((gy * grid + gx) * 3 + a) * 85;
                if (v[o + 4] < scoreThreshold) continue; // score = objectness * class probability <= objectness: skips the argmax for almost every anchor
                int cls = 0;
                for (int c = 1; c < 80; c++) if (v[o + 5 + c] > v[o + 5 + cls]) cls = c;
                float score = v[o + 4] * v[o + 5 + cls];
                if (score < scoreThreshold) continue;

                float cx = (Sigmoid(v[o]) * xyScale[h] - 0.5f * (xyScale[h] - 1) + gx) * strides[h];
                float cy = (Sigmoid(v[o + 1]) * xyScale[h] - 0.5f * (xyScale[h] - 1) + gy) * strides[h];
                float w = MathF.Exp(v[o + 2]) * anchors[h][a * 2];
                float hh = MathF.Exp(v[o + 3]) * anchors[h][a * 2 + 1];
                // Letterboxed 416 space -> original image, clipped.
                float x1 = Math.Clamp((cx - w / 2 - padX) / scale, 0, orgW - 1), x2 = Math.Clamp((cx + w / 2 - padX) / scale, 0, orgW - 1);
                float y1 = Math.Clamp((cy - hh / 2 - padY) / scale, 0, orgH - 1), y2 = Math.Clamp((cy + hh / 2 - padY) / scale, 0, orgH - 1);
                if (x2 > x1 && y2 > y1) candidates.Add(new Detection(Coco.Names80[cls], score, x1, y1, x2, y2));
            }
        }

        // Hard NMS per class.
        List<Detection> kept = [];
        foreach (IGrouping<string, Detection> cls in candidates.GroupBy(d => d.Label))
        {
            List<Detection> pool = cls.OrderByDescending(d => d.Score).ToList();
            while (pool.Count > 0)
            {
                Detection best = pool[0];
                kept.Add(best);
                pool = pool.Skip(1).Where(d => Iou(best, d) <= iouThreshold).ToList();
            }
        }
        return kept;
    }

    static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    static float Iou(Detection a, Detection b)
    {
        float iw = Math.Max(0, Math.Min(a.X2, b.X2) - Math.Max(a.X1, b.X1));
        float ih = Math.Max(0, Math.Min(a.Y2, b.Y2) - Math.Max(a.Y1, b.Y1));
        float inter = iw * ih;
        return inter / ((a.X2 - a.X1) * (a.Y2 - a.Y1) + (b.X2 - b.X1) * (b.Y2 - b.Y1) - inter);
    }
}
