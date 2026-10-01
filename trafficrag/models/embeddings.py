"""CUDA text encoding with length-aware batches and consistent masked pooling."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TextEncoderConfig:
    checkpoint: str = "answerdotai/ModernBERT-base"
    revision: str = "main"
    cache_dir: object = None
    offline: bool = False
    max_length: int = 8192
    batch_size: int = 16
    token_budget: int = 32768

    def __post_init__(self):
        if min(self.max_length, self.batch_size, self.token_budget) < 1:
            raise ValueError("Text encoder lengths and batch budgets must be positive")


def masked_mean(hidden, attention_mask):
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and token masks must align")
    mask = attention_mask[..., None].to(hidden.dtype)
    if (attention_mask.sum(1) == 0).any():
        raise ValueError("Text embeddings require at least one valid token")
    return (hidden * mask).sum(1) / mask.sum(1)


def length_batches(lengths, batch_size, token_budget):
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batch = []
    maximum = 0
    for index in order:
        length = int(lengths[index])
        if length < 1:
            raise ValueError("Encoded captions must have nonempty token sequences")
        if length > token_budget:
            raise ValueError("A caption exceeds the configured per-batch token budget")
        projected = max(maximum, length)
        if batch and (len(batch) >= batch_size or projected * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
            maximum = 0
        batch.append(index)
        maximum = max(maximum, length)
    if batch:
        yield batch


class TextEncoder:
    def __init__(self, config=None, device="cuda"):
        self.config = config or TextEncoderConfig()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Traffic caption embeddings require CUDA")
        self.model = None
        self.tokenizer = None
        self.last_batches = []

    def configure(self):
        from transformers import AutoModel, AutoTokenizer

        options = {
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.offline,
            "revision": self.config.revision,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint, **options)
        self.model = (
            AutoModel.from_pretrained(self.config.checkpoint, attn_implementation="sdpa", **options)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        limit = getattr(self.model.config, "max_position_embeddings", self.config.max_length)
        if self.config.max_length > limit:
            raise ValueError("Text context length exceeds the encoder position capacity")

    def identity(self):
        return {
            "text_encoder": self.config.checkpoint,
            "revision": self.config.revision,
            "max_length": self.config.max_length,
            "pooling": "attention-mask mean",
        }

    @torch.inference_mode()
    def __call__(self, captions):
        captions = list(captions)
        if not captions or any(not isinstance(text, str) or not text.strip() for text in captions):
            raise ValueError("Caption encoding requires nonempty strings")
        if self.model is None:
            self.configure()
        tokenized = self.tokenizer(
            captions, truncation=True, max_length=self.config.max_length, padding=False
        )
        lengths = [len(values) for values in tokenized["input_ids"]]
        output = [None] * len(captions)
        self.last_batches = []
        for indices in length_batches(lengths, self.config.batch_size, self.config.token_budget):
            records = [
                {name: values[index] for name, values in tokenized.items()} for index in indices
            ]
            batch = self.tokenizer.pad(records, padding=True, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            pooled = masked_mean(hidden, batch["attention_mask"]).float()
            if not torch.isfinite(pooled).all() or (pooled.norm(dim=1) == 0).any():
                raise RuntimeError("Text encoder produced a zero or nonfinite representation")
            values = pooled.cpu().numpy()
            for index, embedding in zip(indices, values):
                output[index] = embedding
            self.last_batches.append(
                {
                    "captions": len(indices),
                    "padded_tokens": int(batch["input_ids"].numel()),
                    "valid_tokens": int(batch["attention_mask"].sum()),
                    "longest_caption": max(lengths[index] for index in indices),
                }
            )
        return np.stack(output)

    def encoding_report(self):
        valid = sum(row["valid_tokens"] for row in self.last_batches)
        padded = sum(row["padded_tokens"] for row in self.last_batches)
        return {
            "identity": self.identity(),
            "batches": self.last_batches,
            "valid_tokens": valid,
            "padded_tokens": padded,
            "padding_fraction": 1 - valid / padded if padded else None,
        }


def pretrained_inventory(directory, verify=False):
    import hashlib
    import json
    from pathlib import Path

    root = Path(directory).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    indexes = [root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json"]
    indexed = next((path for path in indexes if path.is_file()), None)
    weight_map = {}
    if indexed:
        weight_map = json.loads(indexed.read_text())["weight_map"]
        if not weight_map:
            raise ValueError("Pretrained weight index contains no tensor assignments")
        filenames = sorted(set(weight_map.values()))
    else:
        filename = next(
            (
                name
                for name in ("model.safetensors", "pytorch_model.bin")
                if (root / name).is_file()
            ),
            None,
        )
        if filename is None:
            raise FileNotFoundError(f"No Transformers weight file or index in {root}")
        filenames = [filename]
    shards = []
    for filename in filenames:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Pretrained index contains an invalid weight path")
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Pretrained tensor shard is absent or empty: {path}")
        item = {"file": filename, "bytes": path.stat().st_size}
        if verify:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            item["sha256"] = digest.hexdigest()
        shards.append(item)
    tokenizer = (root / "tokenizer.json").is_file() or (root / "tokenizer.model").is_file()
    tokenizer = tokenizer or ((root / "vocab.json").is_file() and (root / "merges.txt").is_file())
    if not tokenizer:
        raise FileNotFoundError(f"Tokenizer vocabulary is missing from {root}")
    return {
        "directory": str(root),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "indexed_tensors": len(weight_map) if weight_map else None,
        "shards": shards,
        "weight_bytes": sum(row["bytes"] for row in shards),
        "tokenizer": True,
        "processor_files": sorted(path.name for path in root.glob("*processor*.json")),
    }


def prepare_pretrained_models(
    output, vlm, text_encoder, revision="main", cache_dir=None, offline=False, verify=False
):
    import json
    import shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download, snapshot_download
    from trafficrag.models.motion import VideoMAEEncoder
    from trafficrag.retrieval.storage import atomic_json, file_hash

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "preparation.json"
    identity = {"vlm": vlm, "text_encoder": text_encoder, "revision": revision}
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Pretrained preparation model identity differs from the existing directory"
        )
    if not identity_path.exists():
        for name, source in (("vlm", vlm), ("text_encoder", text_encoder)):
            target = root / name
            if target.is_dir() and any(target.iterdir()) and Path(source).resolve() != target:
                raise FileExistsError(
                    "An existing pretrained directory has no preparation identity"
                )
        atomic_json(identity_path, identity)
    inventories = {}
    allowed = ["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.tiktoken", "*.py"]
    for name, source in (("vlm", vlm), ("text_encoder", text_encoder)):
        local = Path(source)
        target = root / name
        if local.is_dir():
            source_root = local.resolve()
            if source_root != target:
                if target.exists() and any(target.iterdir()):
                    raise FileExistsError(f"Local pretrained destination must be empty: {target}")
                if target.is_relative_to(source_root):
                    raise ValueError("Pretrained destination cannot be inside its source")
                shutil.copytree(
                    source_root,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", ".cache"),
                )
        else:
            snapshot_download(
                source,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(target),
                local_files_only=offline,
                allow_patterns=allowed,
            )
        inventory = pretrained_inventory(target, verify)
        inventory.update(source=source, revision=revision)
        inventories[name] = inventory
    encoder = VideoMAEEncoder.Config()
    motion_directory = root / "motion"
    motion_path = motion_directory / encoder.filename
    if not motion_path.is_file():
        cached = hf_hub_download(
            encoder.repository,
            encoder.filename,
            revision=encoder.revision,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, motion_path)
    if motion_path.stat().st_size == 0:
        raise ValueError("Prepared VideoMAE weight file is empty")
    inventories["motion"] = {
        "source": encoder.repository,
        "revision": encoder.revision,
        "weights": str(motion_path),
        "bytes": motion_path.stat().st_size,
    }
    if verify:
        inventories["motion"]["sha256"] = file_hash(motion_path)
    language_settings = {
        "vlm": str(root / "vlm"),
        "text_encoder": str(root / "text_encoder"),
        "revision": revision,
        "offline": True,
    }
    atomic_json(root / "language.json", language_settings)
    atomic_json(root / "inventory.json", inventories)
    print(json.dumps({"output": str(root), "models": list(inventories)}, indent=2))
    return inventories
