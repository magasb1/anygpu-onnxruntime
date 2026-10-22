// RF-DETR output decoding, shared by RfDetr.Example and LiveDetection.Example (linked with a Compile Include).
using Microsoft.ML.OnnxRuntime.Tensors;

static class RfDetrDecoder
{
    // Outputs: boxes [N, 300, 4] as normalised cx, cy, w, h and logits [N, 300, 91] (original COCO category ids). Picked by shape, not name.
    // Decodes image `index` of the batch: sigmoid on the logits, keep queries above the threshold. No NMS: DETR-style models do not need it.
    public static List<Detection> Decode(Tensor<float>[] outputs, int index, int orgW, int orgH)
    {
        const float threshold = 0.5f;
        DenseTensor<float> boxTensor = (DenseTensor<float>)outputs.First(t => t.Dimensions[2] == 4);
        DenseTensor<float> logitTensor = (DenseTensor<float>)outputs.First(t => t.Dimensions[2] != 4);
        int queries = boxTensor.Dimensions[1], classes = logitTensor.Dimensions[2];
        string[] names = classes == Coco.Names91.Length ? Coco.Names91 : Coco.Names80;
        // Flat spans over this image; Tensor's [i, j, k] indexer allocates its index array on every access.
        ReadOnlySpan<float> boxes = boxTensor.Buffer.Span.Slice(index * queries * 4, queries * 4);
        ReadOnlySpan<float> logits = logitTensor.Buffer.Span.Slice(index * queries * classes, queries * classes);

        List<Detection> kept = [];
        for (int q = 0; q < queries; q++)
        {
            ReadOnlySpan<float> l = logits.Slice(q * classes, classes);
            int cls = 0;
            for (int c = 1; c < classes; c++) if (l[c] > l[cls]) cls = c;
            float score = 1f / (1f + MathF.Exp(-l[cls]));
            if (score < threshold) continue;
            ReadOnlySpan<float> b = boxes.Slice(q * 4, 4);
            float cx = b[0] * orgW, cy = b[1] * orgH, w = b[2] * orgW, h = b[3] * orgH;
            kept.Add(new Detection(names[cls], score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2));
        }
        return kept;
    }
}
