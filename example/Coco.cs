// COCO class names, shared by the detection examples (linked into each project with a Compile Include).
static class Coco
{
    // The original 91 COCO category ids (1-based, with the 11 unused slots as "N/A"), as RF-DETR's COCO checkpoints emit them.
    public static readonly string[] Names91 =
    [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    ];

    // The 80 trained classes in order (0-based), as YOLO models emit them.
    public static readonly string[] Names80 = Names91.Where(n => n != "N/A").ToArray();
}

record Detection(string Label, float Score, float X1, float Y1, float X2, float Y2)
{
    public override string ToString() => $"{Label,-14} {Score:F2}  [{X1:F0},{Y1:F0},{X2:F0},{Y2:F0}]";
}
