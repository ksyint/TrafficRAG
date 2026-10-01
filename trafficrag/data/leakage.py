"""Video, group, and content overlap across motion, memory, and evaluation data."""

from collections import defaultdict
from pathlib import Path

from trafficrag.retrieval.storage import file_hash


def recording_identities(manifest, hashes=False):
    records = {}
    for row in manifest:
        path = Path(row.video).resolve()
        if row.video_id not in records:
            records[row.video_id] = {
                "video_id": row.video_id,
                "group": row.group,
                "path": str(path),
                "bytes": path.stat().st_size,
                "domains": set(),
                "records": [],
            }
            if hashes:
                records[row.video_id]["sha256"] = file_hash(path)
        record = records[row.video_id]
        if record["path"] != str(path) or record["group"] != row.group:
            raise ValueError("Recording identity changed within a manifest")
        record["domains"].add(row.domain)
        record["records"].append(row.identifier)
    for record in records.values():
        record["domains"] = sorted(record["domains"])
    return records


def overlap_report(partitions, hashes=False):
    identities = {
        name: recording_identities(manifest, hashes) for name, manifest in partitions.items()
    }
    fields = ["video_id", "group", "path"] + (["sha256"] if hashes else [])
    reports = {}
    for field in fields:
        ownership = defaultdict(list)
        for split, values in identities.items():
            for record in values.values():
                ownership[record[field]].append(
                    {"split": split, "video_id": record["video_id"], "path": record["path"]}
                )
        reports[field] = [
            {"identity": identity, "occurrences": occurrences}
            for identity, occurrences in ownership.items()
            if len({row["split"] for row in occurrences}) > 1
        ]
    return {
        "partitions": {
            name: {
                "videos": len(values),
                "records": sum(len(row["records"]) for row in values.values()),
            }
            for name, values in identities.items()
        },
        "overlap": reports,
        "disjoint": not any(reports.values()),
    }


def require_disjoint(partitions, hashes=False):
    result = overlap_report(partitions, hashes)
    if not result["disjoint"]:
        fields = [field for field, rows in result["overlap"].items() if rows]
        raise ValueError("Data partitions share " + ", ".join(fields))
    return result


def conflicting_segments(manifest, tolerance=1e-6):
    groups = defaultdict(list)
    for row in manifest:
        if row.start is not None and row.label is not None:
            groups[(row.video_id, row.domain)].append(row)
    conflicts = []
    for (video, domain), rows in groups.items():
        rows.sort(key=lambda row: row.start)
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if right.start >= left.end - tolerance:
                    break
                overlap = min(left.end, right.end) - right.start
                identical = (
                    abs(left.start - right.start) <= tolerance
                    and abs(left.end - right.end) <= tolerance
                )
                if left.label != right.label and identical:
                    conflicts.append(
                        {
                            "video_id": video,
                            "domain": domain,
                            "left_id": left.identifier,
                            "right_id": right.identifier,
                            "overlap_seconds": overlap,
                            "labels": [left.label, right.label],
                        }
                    )
    return conflicts


def source_summary(partitions, hashes=False):
    result = overlap_report(partitions, hashes)
    result["label_conflicts"] = {
        name: conflicting_segments(manifest) for name, manifest in partitions.items()
    }
    return result
