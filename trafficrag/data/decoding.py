"""Video presentation-time inspection and deterministic frame sampling records."""

from pathlib import Path

import av
import numpy as np


def inspect_video(path, scan=False):
    path = Path(path).resolve()
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
            if container.duration
            else None
        )
        report = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
            "base_rate": float(stream.base_rate) if stream.base_rate else None,
            "time_base": str(stream.time_base),
            "origin_seconds": origin,
            "duration_seconds": duration,
            "declared_frames": stream.frames,
        }
        if scan:
            timestamps = []
            missing = 0
            for frame in container.decode(stream):
                if frame.time is None:
                    missing += 1
                else:
                    timestamps.append(float(frame.time) - origin)
            intervals = np.diff(timestamps)
            report["decoded"] = {
                "frames": len(timestamps) + missing,
                "missing_timestamps": missing,
                "nonincreasing_timestamps": int(np.sum(intervals <= 0)),
                "first_seconds": timestamps[0] if timestamps else None,
                "last_seconds": timestamps[-1] if timestamps else None,
                "median_frame_seconds": float(np.median(intervals)) if len(intervals) else None,
                "minimum_frame_seconds": float(intervals.min()) if len(intervals) else None,
                "maximum_frame_seconds": float(intervals.max()) if len(intervals) else None,
            }
    return report


def sample_report(source, start, end, count, resize=None):
    frames, relative = source.read(start, end, count, resize)
    absolute = relative + start
    return {
        "source": source.path,
        "crop": [start, end],
        "requested_frames": count,
        "shape": list(frames.shape),
        "relative_timestamps": relative.tolist(),
        "absolute_timestamps": absolute.tolist(),
        "unique_timestamps": len(np.unique(relative)),
        "duplicate_fraction": 1 - len(np.unique(relative)) / len(relative),
        "first_offset_seconds": float(relative[0]),
        "last_margin_seconds": float(end - absolute[-1]),
        "pixel_minimum": int(frames.min()),
        "pixel_maximum": int(frames.max()),
    }


def timestamp_metadata(relative, duration):
    timestamps = np.asarray(relative, dtype=float)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("Video language inputs require at least two timestamps")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("Video timestamps must be finite and monotonic")
    if timestamps[0] < 0 or timestamps[-1] >= duration + 1e-6:
        raise ValueError("Video timestamps must remain inside the crop")
    return {
        "fps": 1000.0,
        "total_num_frames": max(1, int(np.ceil(duration * 1000))),
        "frames_indices": np.round(timestamps * 1000).astype(int).tolist(),
    }


def manifest_video_report(manifest, scan=False):
    reports = {}
    for row in manifest:
        if row.video_id not in reports:
            reports[row.video_id] = inspect_video(row.video, scan)
        duration = reports[row.video_id]["duration_seconds"]
        if duration is None or duration <= 0:
            raise ValueError(f"Video has no positive duration: {row.video_id}")
        if row.end is not None and row.end > duration + 1e-6:
            raise ValueError(f"Annotated segment exceeds video duration: {row.identifier}")
        if row.target is not None and row.target[1] > duration + 1e-6:
            raise ValueError(f"Target interval exceeds video duration: {row.identifier}")
    return reports
