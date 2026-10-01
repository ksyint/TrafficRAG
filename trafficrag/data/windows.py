"""Motion training windows derived from full-video temporal annotations."""

from dataclasses import dataclass

from trafficrag.grounding import Interval, segment_video, temporal_iou
from trafficrag.data.records import VideoRecord, VideoManifest, finite_interval


@dataclass(frozen=True)
class WindowLabeling:
    window: float = 2.0
    stride: float = 1.0
    positive_overlap: float = 0.5
    negative_overlap: float = 0.0
    keep_ambiguous: bool = False

    def __post_init__(self):
        if not 0 < self.stride <= self.window:
            raise ValueError("Motion windows require 0 < stride <= window")
        if not 0 <= self.negative_overlap < self.positive_overlap <= 1:
            raise ValueError("Negative overlap must be below the positive threshold")


def overlap_fraction(window, target):
    if target is None:
        return 0.0
    intersection = max(0.0, min(window.end, target.end) - max(window.start, target.start))
    return intersection / (window.end - window.start)


def label_window(window, target, settings):
    overlap = overlap_fraction(window, target)
    if overlap >= settings.positive_overlap:
        return 1, overlap
    if overlap <= settings.negative_overlap:
        return 0, overlap
    return (int(overlap >= 0.5), overlap) if settings.keep_ambiguous else (None, overlap)


def windows_for_record(record, duration, settings=None):
    settings = settings or WindowLabeling()
    finite_interval(record.target, duration)
    target = Interval(*record.target) if record.target is not None else None
    result, ignored = [], []
    for number, interval in enumerate(segment_video(duration, settings.window, settings.stride)):
        label, overlap = label_window(interval, target, settings)
        identity = f"{record.identifier}:window-{number:05d}"
        detail = {
            "source_record": record.identifier,
            "target_overlap_fraction": overlap,
            "target_iou": temporal_iou(interval, target),
            "video_duration": duration,
        }
        if label is None:
            ignored.append(dict(id=identity, start=interval.start, end=interval.end, **detail))
            continue
        result.append(
            VideoRecord(
                identity,
                record.video_id,
                record.video,
                record.domain,
                record.target,
                record.group,
                label,
                interval.start,
                interval.end,
                detail,
            )
        )
    return result, ignored


def annotation_windows(manifest, duration_of, settings=None):
    settings = settings or WindowLabeling()
    rows, omitted = [], []
    seen = set()
    for record in manifest:
        key = record.video_id, record.domain
        if key in seen:
            raise ValueError(
                "Window preparation needs one full-video annotation per video and domain"
            )
        seen.add(key)
        if record.start is not None or record.end is not None:
            raise ValueError("Use full-video annotations rather than existing segment rows")
        segments, ambiguous = windows_for_record(record, duration_of(record.video), settings)
        rows.extend(segments)
        omitted.extend(ambiguous)
    if not rows:
        raise ValueError("No labeled motion windows were produced")
    return VideoManifest(rows), omitted


def window_summary(manifest):
    domains = {}
    for record in manifest:
        if record.start is None or record.label is None:
            raise ValueError("Window summaries require labeled temporal segments")
        group = domains.setdefault(
            record.domain, {"positive": 0, "negative": 0, "seconds": 0.0, "videos": set()}
        )
        group["positive" if record.label else "negative"] += 1
        group["seconds"] += record.end - record.start
        group["videos"].add(record.video_id)
    for values in domains.values():
        values["videos"] = len(values["videos"])
        total = values["positive"] + values["negative"]
        values["positive_fraction"] = values["positive"] / total
        values["mean_duration"] = values["seconds"] / total
    return domains


def temporal_coverage(intervals, duration):
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    merged = []
    for start, end in ordered:
        finite_interval([start, end], duration)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    return {"seconds": covered, "fraction": covered / duration, "intervals": merged}
