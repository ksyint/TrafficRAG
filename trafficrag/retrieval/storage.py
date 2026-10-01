"""Versioned caption-memory bundles with embedding shards and integrity records."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from trafficrag.retrieval.records import MemoryRecords


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_memory(directory, records, embeddings, encoder, shard_rows=10000):
    root = Path(directory)
    if (root / "memory.json").exists():
        raise FileExistsError("A completed memory bundle already exists at this destination")
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(records) or not np.isfinite(values).all():
        raise ValueError("Memory embeddings must be a finite matrix aligned to records")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0) or shard_rows < 1:
        raise ValueError("Embedding norms and shard row counts must be positive")
    values = values / norms[:, None]
    root.mkdir(parents=True, exist_ok=True)
    records.write(root / "captions.jsonl")
    shards = []
    for number, start in enumerate(range(0, len(values), shard_rows)):
        path = root / f"embeddings-{number:05d}.npy"
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.save(stream, values[start : start + shard_rows], allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        shards.append(
            {
                "path": path.name,
                "start": start,
                "rows": min(shard_rows, len(values) - start),
                "sha256": file_hash(path),
            }
        )
    metadata = {
        "format": "traffic-caption-memory-v1",
        "domain": records.domain,
        "entries": len(records),
        "dimension": values.shape[1],
        "dtype": "float32",
        "normalization": "l2",
        "encoder": dict(encoder),
        "records": {"path": "captions.jsonl", "sha256": file_hash(root / "captions.jsonl")},
        "shards": shards,
        "summary": records.summary(),
    }
    atomic_json(root / "memory.json", metadata)
    return metadata


def safe_member(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Memory bundle members must stay inside the bundle")
    result = Path(root) / path
    if result.is_symlink() or not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Memory bundle member escapes its root")
    return result


def read_memory(directory, verify=True):
    root = Path(directory)
    metadata = json.loads((root / "memory.json").read_text())
    if metadata.get("format") != "traffic-caption-memory-v1":
        raise ValueError("Unsupported caption memory format")
    records_path = safe_member(root, metadata["records"]["path"])
    if verify and file_hash(records_path) != metadata["records"]["sha256"]:
        raise ValueError("Caption records differ from their saved digest")
    records = MemoryRecords.read(records_path)
    arrays = []
    cursor = 0
    for shard in metadata["shards"]:
        if shard["start"] != cursor:
            raise ValueError("Embedding shards contain a row gap or overlap")
        path = safe_member(root, shard["path"])
        if verify and file_hash(path) != shard["sha256"]:
            raise ValueError(f"Embedding shard differs from its saved digest: {path}")
        values = np.load(path, allow_pickle=False)
        if values.shape != (shard["rows"], metadata["dimension"]) or values.dtype != np.float32:
            raise ValueError("Embedding shard shape or precision differs from the manifest")
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values, axis=1), 1, atol=1e-5
        ):
            raise ValueError("Memory embeddings must be finite unit vectors")
        arrays.append(values)
        cursor += len(values)
    if (
        cursor != metadata["entries"]
        or cursor != len(records)
        or records.domain != metadata["domain"]
    ):
        raise ValueError("Caption memory dimensions or domain are inconsistent")
    return records, np.concatenate(arrays), metadata


def export_legacy(directory, output):
    records, vectors, metadata = read_memory(directory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez(
            stream,
            embeddings=vectors,
            labels=np.asarray([row.label for row in records]),
            captions=np.asarray([row.caption for row in records]),
            ids=np.asarray([row.identifier for row in records]),
        )
    atomic_json(
        output.with_suffix(".models.json"),
        {
            "domain": records.domain,
            "models": metadata["encoder"],
            "pooling": "attention-mask mean",
            "similarity": "cosine",
        },
    )


def encoder_signature(metadata):
    encoder = metadata["encoder"]
    required = ("text_encoder", "revision", "max_length", "pooling")
    missing = [key for key in required if key not in encoder]
    if missing:
        raise ValueError(f"Caption memory lacks an encoder specification: {missing}")
    return {key: encoder[key] for key in required}


def merge_memories(directories, output, shard_rows=10000):
    if len(directories) < 2:
        raise ValueError("Memory merge requires at least two input bundles")
    sources = [Path(path).resolve() for path in directories]
    destination = Path(output).resolve()
    if len(set(sources)) != len(sources) or destination in sources:
        raise ValueError("Memory merge inputs and destination must use distinct directories")
    rows = []
    vectors = []
    lookup = {}
    duplicates = []
    provenance = []
    domain = signature = dimension = encoder = None
    for source in sources:
        records, values, metadata = read_memory(source)
        current = encoder_signature(metadata)
        if signature is None:
            domain = records.domain
            signature = current
            dimension = values.shape[1]
            encoder = metadata["encoder"]
        elif records.domain != domain or current != signature or values.shape[1] != dimension:
            raise ValueError(
                "Merged memories must share domain, encoder, revision, and embedding width"
            )
        accepted = 0
        for index, row in enumerate(records):
            if row.identifier in lookup:
                previous = lookup[row.identifier]
                if rows[previous] != row or not np.allclose(
                    vectors[previous], values[index], atol=1e-6
                ):
                    raise ValueError(f"Conflicting memory entry: {row.identifier}")
                duplicates.append({"id": row.identifier, "source": str(source)})
                continue
            lookup[row.identifier] = len(rows)
            rows.append(row)
            vectors.append(values[index])
            accepted += 1
        provenance.append(
            {
                "directory": str(source),
                "manifest_sha256": file_hash(source / "memory.json"),
                "entries": len(records),
                "accepted_entries": accepted,
            }
        )
    result = save_memory(destination, MemoryRecords(rows), np.stack(vectors), encoder, shard_rows)
    atomic_json(
        destination / "provenance.json",
        {
            "operation": "merge",
            "sources": provenance,
            "identical_duplicates": duplicates,
            "entries": len(rows),
        },
    )
    return result


def subset_memory(directory, output, maximum_rows, seed=42, maximum_per_video=None, balanced=False):
    from trafficrag.retrieval.records import select_memory_rows

    source = Path(directory).resolve()
    destination = Path(output).resolve()
    if source == destination:
        raise ValueError("Memory subsets require a separate destination")
    records, vectors, metadata = read_memory(source)
    indices, chosen, report = select_memory_rows(
        records, maximum_rows, seed, maximum_per_video, balanced
    )
    result = save_memory(destination, chosen, vectors[indices], metadata["encoder"])
    report.update(
        operation="subset", source=str(source), source_sha256=file_hash(source / "memory.json")
    )
    atomic_json(destination / "selection.json", report)
    return result
