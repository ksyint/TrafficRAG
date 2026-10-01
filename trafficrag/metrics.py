def temporal_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union
