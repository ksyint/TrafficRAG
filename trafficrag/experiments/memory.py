"""Caption memory construction, retrieval diagnostics, and portable exports."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trafficrag.data.records import DOMAINS, VideoManifest
from trafficrag.models.motion import VideoSource, execution_device
from trafficrag.models.backends import VideoLanguageSystem
from trafficrag.models.embeddings import TextEncoder, TextEncoderConfig
from trafficrag.models.prompts import prompt_identity
from trafficrag.grounding import Interval
from trafficrag.retrieval.records import MemoryRecord, MemoryRecords
from trafficrag.retrieval.cache import TextEmbeddingCache
from trafficrag.retrieval.storage import atomic_json, save_memory, read_memory, export_legacy
from trafficrag.retrieval.metrics import memory_diagnostics, caption_conflicts


class CaptionJournal:
    def __init__(self, directory, identity, identifiers, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.directory / "caption-session.json"
        self.path = self.directory / "captions.jsonl"
        self.identifiers = set(identifiers)
        if self.identity_path.exists():
            if not resume:
                raise FileExistsError("Caption construction already exists, use --resume")
            if json.loads(self.identity_path.read_text()) != identity:
                raise ValueError("Caption model, source data, or prompt settings changed")
        else:
            if self.path.exists():
                raise ValueError("Caption rows exist without their session identity")
            atomic_json(self.identity_path, identity)
        self.rows = {}
        if self.path.exists():
            with self.path.open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.endswith("\n"):
                        raise ValueError(
                            f"Caption journal has an incomplete final record at line {number}"
                        )
                    row = MemoryRecord.from_dict(json.loads(line))
                    if row.identifier in self.rows or row.identifier not in self.identifiers:
                        raise ValueError("Caption journal contains duplicate or unknown IDs")
                    self.rows[row.identifier] = row

    def append(self, row):
        import os

        if row.identifier in self.rows or row.identifier not in self.identifiers:
            raise ValueError("Caption identity is duplicated or outside the source manifest")
        with self.path.open("a") as stream:
            stream.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[row.identifier] = row

    def ordered(self, identifiers):
        if set(identifiers) != set(self.rows):
            raise ValueError("Caption construction has not completed all source rows")
        return MemoryRecords([self.rows[identity] for identity in identifiers])


def caption_manifest(manifest, output, config, device="cuda", resume=False):
    ids = [row.identifier for row in manifest]
    domains = {row.domain for row in manifest}
    if len(domains) != 1:
        raise ValueError("Caption construction requires exactly one traffic domain")
    domain = next(iter(domains))
    for row in manifest:
        if row.start is None or row.label is None:
            raise ValueError("Knowledge-base records need annotated start, end, and binary label")
    identity = {
        "format": "traffic-caption-build-v1",
        "manifest": manifest.digest(include_sources=True),
        "language": asdict(config),
        "prompt": prompt_identity(domain),
    }
    journal = CaptionJournal(output, identity, ids, resume)
    system = VideoLanguageSystem(config, device)
    videos = {}
    for row in manifest:
        if row.identifier in journal.rows:
            continue
        caption = row.metadata.get("caption")
        if caption is None:
            if row.video_id not in videos:
                videos[row.video_id] = VideoSource(row.video)
            source = videos[row.video_id]
            if row.end > source.duration + 1e-6:
                raise ValueError("Memory segment extends beyond its source video")
            caption = system.caption(source, Interval(row.start, row.end), domain)
        journal.append(
            MemoryRecord(
                row.identifier,
                row.video_id,
                row.domain,
                caption,
                row.label,
                row.start,
                row.end,
                row.group,
            )
        )
    del system
    return journal.ordered(ids)


def memory_build_cli():
    parser = argparse.ArgumentParser(
        description="Build a portable CUDA caption memory with resumable captions."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--captions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--vlm", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-encoder", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=10000)
    parser.add_argument("--exclude-manifests", nargs="*", default=[])
    args = parser.parse_args()
    device = execution_device(args.device)
    output = Path(args.output)
    if (output / "memory.json").exists():
        raise FileExistsError("Completed memory bundle already exists")
    if args.manifest:
        manifest = VideoManifest.read(args.manifest, args.domain)
        if args.exclude_manifests:
            from trafficrag.data.leakage import require_disjoint

            partitions = {"memory": manifest}
            for index, path in enumerate(args.exclude_manifests):
                partitions[f"excluded-{index}"] = VideoManifest.read(path)
            # Excluded collections may overlap each other, but none may overlap memory.
            for key in list(partitions)[1:]:
                require_disjoint({"memory": manifest, key: partitions[key]})
        language = VideoLanguageSystem.Config(
            vlm=args.vlm,
            text_encoder=args.text_encoder,
            revision=args.revision,
            cache_dir=args.cache_dir or "",
            offline=args.offline,
        )
        records = caption_manifest(manifest, output / "caption-work", language, device, args.resume)
    else:
        records = MemoryRecords.read(args.captions)
        if args.domain and records.domain != args.domain:
            raise ValueError("Caption domain differs from --domain")
        for path in args.exclude_manifests:
            excluded = VideoManifest.read(path)
            ids = {row.video_id for row in excluded}
            groups = {row.group for row in excluded}
            if ids & set(records.by_video) or groups & set(records.by_group):
                raise ValueError("Caption memory overlaps an excluded video collection")
    encoder = TextEncoder(
        TextEncoderConfig(
            checkpoint=args.text_encoder,
            revision=args.revision,
            cache_dir=args.cache_dir,
            offline=args.offline,
            batch_size=args.batch_size,
        ),
        device,
    )
    cache = TextEmbeddingCache(
        args.embedding_cache or output / "embedding-cache", encoder.identity()
    )
    embeddings = cache.encode([row.caption for row in records], encoder, args.batch_size)
    identity = dict(encoder.identity(), vlm=args.vlm)
    report = save_memory(output, records, embeddings, identity, args.shard_rows)
    atomic_json(
        output / "encoding.json", {"cache": cache.summary(), "encoder": encoder.encoding_report()}
    )
    print(json.dumps(report["summary"], indent=2))


def memory_audit_cli():
    parser = argparse.ArgumentParser(
        description="Inspect memory integrity and leave-video-out retrieval quality."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=(1, 3, 5, 10))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    records, vectors, metadata = read_memory(args.memory)
    report = {"metadata": metadata, "caption_conflicts": caption_conflicts(records)}
    if not args.metadata_only:
        report["retrieval"] = memory_diagnostics(
            records, vectors, args.ks, args.device, args.batch_size
        )
    atomic_json(args.output, report)
    print(json.dumps({"entries": len(records), "output": args.output}, indent=2))


def memory_export_cli():
    parser = argparse.ArgumentParser(
        description="Export a memory bundle for existing NPZ-based grounding runs."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_legacy(args.memory, args.output)
    print(json.dumps({"output": args.output}, indent=2))


def memory_merge_cli():
    from trafficrag.retrieval.storage import merge_memories

    parser = argparse.ArgumentParser(
        description="Combine compatible caption memories with identity checks."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-rows", type=int, default=10000)
    args = parser.parse_args()
    metadata = merge_memories(args.inputs, args.output, args.shard_rows)
    print(json.dumps(metadata["summary"], indent=2))


def memory_subset_cli():
    from trafficrag.retrieval.storage import subset_memory

    parser = argparse.ArgumentParser(
        description="Build a reproducible caption-memory budget by source video."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-rows", type=int, required=True)
    parser.add_argument("--maximum-per-video", type=int)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = subset_memory(
        args.memory,
        args.output,
        args.maximum_rows,
        args.seed,
        args.maximum_per_video,
        args.balanced,
    )
    print(json.dumps(metadata["summary"], indent=2))


def retrieval_query_cli():
    from trafficrag.retrieval.metrics import evaluate_queries
    from trafficrag.retrieval.storage import encoder_signature

    parser = argparse.ArgumentParser(
        description="Evaluate held-out captions against a frozen CUDA memory index."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=(1, 3, 5, 10))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    records, vectors, metadata = read_memory(args.memory)
    queries = MemoryRecords.read(args.queries)
    identity = encoder_signature(metadata)
    encoder = TextEncoder(
        TextEncoderConfig(
            checkpoint=identity["text_encoder"],
            revision=identity["revision"],
            max_length=identity["max_length"],
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            offline=args.offline,
        ),
        execution_device(args.device),
    )
    cache = TextEmbeddingCache(
        args.embedding_cache or Path(args.output).parent / "query-embeddings", identity
    )
    embeddings = cache.encode([row.caption for row in queries], encoder, args.batch_size)
    report = evaluate_queries(
        records, vectors, queries, embeddings, args.ks, args.device, args.batch_size
    )
    report["encoding"] = encoder.encoding_report()
    report["embedding_cache"] = cache.summary()
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "queries": len(queries),
                "cutoffs": [
                    {"k": row["used_k"], "accuracy": row["classification"]["accuracy"]}
                    for row in report["cutoffs"]
                ],
            },
            indent=2,
        )
    )
