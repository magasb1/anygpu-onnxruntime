// Non-maximum suppression, shared by the YOLO decoders (linked with a Compile Include).
static class Nms
{
    // Hard NMS per class: keeps the best box of each cluster, drops the boxes of the same class overlapping it by more than iouThreshold.
    public static List<Detection> PerClass(List<Detection> candidates, float iouThreshold)
    {
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

    static float Iou(Detection a, Detection b)
    {
        float iw = Math.Max(0, Math.Min(a.X2, b.X2) - Math.Max(a.X1, b.X1));
        float ih = Math.Max(0, Math.Min(a.Y2, b.Y2) - Math.Max(a.Y1, b.Y1));
        float inter = iw * ih;
        return inter / ((a.X2 - a.X1) * (a.Y2 - a.Y1) + (b.X2 - b.X1) * (b.Y2 - b.Y1) - inter);
    }
}
