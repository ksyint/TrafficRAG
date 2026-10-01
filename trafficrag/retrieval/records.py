"""Typed caption-memory rows with stable source-video identities."""

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from trafficrag.data.records import DOMAINS, finite_interval


@dataclass(frozen=True)
class MemoryRecord:
    identifier: str
    video_id: str
    domain: str
    caption: str
    label: int
    start: float
    end: float
    group: str = ""

    def __post_init__(self):
        if not self.identifier or not self.video_id or self.domain not in DOMAINS:
            raise ValueError("Memory rows require a segment ID, video ID, and known domain")
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValueError("Memory captions must be nonempty strings")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("Memory labels must be binary integers")
        if not self.group:
            object.__setattr__(self, "group", self.video_id)
        finite_interval([self.start, self.end])

    @classmethod
    def from_dict(cls, row):
        return cls(
            identifier=str(row.get("id", row.get("identifier", ""))),
            video_id=str(row["video_id"]),
            domain=row["domain"],
            caption=row["caption"],
            label=row["label"],
            start=float(row["start"]),
            end=float(row["end"]),
            group=str(row.get("group", row["video_id"])),
        )

    def as_dict(self):
        values = asdict(self)
        values["id"] = values.pop("identifier")
        return values


class MemoryRecords:
    def __init__(self, rows):
        self.rows = tuple(rows)
        if not self.rows:
            raise ValueError("Caption memory must contain entries")
        ids = [row.identifier for row in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Caption memory IDs must be unique")
        domains = {row.domain for row in self.rows}
        if len(domains) != 1:
            raise ValueError("Build independent caption memory for each traffic domain")
        self.domain = domains.pop()
        self.by_id = {row.identifier: index for index, row in enumerate(self.rows)}
        self.by_video = {}
        self.by_group = {}
        for index, row in enumerate(self.rows):
            self.by_video.setdefault(row.video_id, []).append(index)
            self.by_group.setdefault(row.group, []).append(index)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @classmethod
    def read(cls, path):
        rows = []
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(MemoryRecord.from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid caption memory row {line_number}: {error}") from error
        return cls(rows)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row.as_dict(), ensure_ascii=False) + "\n" for row in self)
        )

    def exclusions(self, video_ids=None, groups=None):
        if video_ids is None and groups is None:
            raise ValueError("Retrieval exclusions need query video or group identities")
        count = len(video_ids) if video_ids is not None else len(groups)
        if groups is not None and len(groups) != count:
            raise ValueError("Query video and group identities must align")
        excluded = []
        for index in range(count):
            indices = set()
            if video_ids is not None:
                indices.update(self.by_video.get(str(video_ids[index]), []))
            if groups is not None:
                indices.update(self.by_group.get(str(groups[index]), []))
            excluded.append(sorted(indices))
        return excluded

    def summary(self):
        labels = Counter(row.label for row in self)
        captions = Counter(row.caption.strip().casefold() for row in self)
        return {
            "domain": self.domain,
            "entries": len(self),
            "videos": len(self.by_video),
            "groups": len(self.by_group),
            "positive": labels[1],
            "negative": labels[0],
            "positive_fraction": labels[1] / len(self),
            "unique_captions": len(captions),
            "repeated_captions": sum(count - 1 for count in captions.values()),
            "mean_segment_seconds": sum(row.end - row.start for row in self) / len(self),
        }


def select_memory_rows(records, maximum_rows, seed=42, maximum_per_video=None, balanced=False):
    import numpy as np

    if maximum_rows < 1 or (maximum_per_video is not None and maximum_per_video < 1):
        raise ValueError("Memory row and source-video budgets must be positive")
    generator = np.random.default_rng(seed)
    queues = {}
    for video, indices in sorted(records.by_video.items()):
        indices = list(indices)
        generator.shuffle(indices)
        queues[video] = indices[:maximum_per_video] if maximum_per_video else indices
    videos = list(queues)
    generator.shuffle(videos)
    available = []
    while any(queues.values()):
        for video in videos:
            if queues[video]:
                available.append(queues[video].pop())
    budget = min(maximum_rows, len(available))
    if balanced:
        by_label = {
            label: [index for index in available if records.rows[index].label == label]
            for label in (0, 1)
        }
        per_class = min(budget // 2, *(len(indices) for indices in by_label.values()))
        if per_class == 0:
            raise ValueError(
                "Balanced memory selection needs both labels and a budget of at least two"
            )
        selected = by_label[0][:per_class] + by_label[1][:per_class]
    else:
        selected = available[:budget]
    selected = sorted(selected)
    chosen = MemoryRecords([records.rows[index] for index in selected])
    report = {
        "requested_rows": maximum_rows,
        "selected_rows": len(selected),
        "seed": seed,
        "maximum_per_video": maximum_per_video,
        "balanced": balanced,
        "before": records.summary(),
        "after": chosen.summary(),
        "selected_ids": [row.identifier for row in chosen],
    }
    return selected, chosen, report


def memory_overlap(left, right):
    left_captions = {}
    right_captions = {}
    for table, collection in ((left_captions, left), (right_captions, right)):
        for row in collection:
            normalized = " ".join(row.caption.casefold().split())
            table.setdefault(normalized, []).append(row.identifier)
    same_ids = sorted(left.by_id.keys() & right.by_id.keys())
    conflicts = []
    for identity in same_ids:
        before = left.rows[left.by_id[identity]]
        after = right.rows[right.by_id[identity]]
        if before != after:
            conflicts.append(identity)
    return {
        "shared_ids": same_ids,
        "conflicting_ids": conflicts,
        "shared_videos": sorted(left.by_video.keys() & right.by_video.keys()),
        "shared_groups": sorted(left.by_group.keys() & right.by_group.keys()),
        "shared_caption_count": len(left_captions.keys() & right_captions.keys()),
        "shared_captions": [
            {"caption": caption, "left": left_captions[caption], "right": right_captions[caption]}
            for caption in sorted(left_captions.keys() & right_captions.keys())
        ],
    }
