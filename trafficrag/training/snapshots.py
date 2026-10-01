"""Resumable motion checkpoints compatible with the grounding runner."""

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def capture_rng(loader):
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": {
            "algorithm": state[0],
            "keys": state[1].tolist(),
            "position": state[2],
            "gaussian": state[3],
            "cached": state[4],
        },
        "loader": loader.generator.get_state() if loader.generator is not None else None,
    }


def restore_rng(state, loader):
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["gaussian"],
            numpy_state["cached"],
        )
    )
    if state["loader"] is not None:
        if loader.generator is None:
            raise ValueError("Resume state requires a DataLoader generator")
        loader.generator.set_state(state["loader"].cpu())


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_motion_snapshot(path, trainer, epoch, best, history):
    system = trainer.system
    state = {
        "format": "traffic-motion-training-v1",
        "model": {name: value.detach().cpu() for name, value in system.model.state_dict().items()},
        "feature_dim": system.feature_dim,
        "encoder_config": asdict(system.encoder_cfg),
        "domain": system.domain,
        "config": asdict(system.cfg),
        "training": asdict(trainer.options),
        "objective": asdict(trainer.objective),
        "optimizer": system.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "rng": capture_rng(trainer.train_loader),
        "sampler": trainer.train_loader.sampler.state_dict(),
        "epoch": epoch,
        "best": best,
        "history": history,
        "train_digest": trainer.train_digest,
        "validation_digest": trainer.validation_digest,
    }
    atomic_checkpoint(path, state)


def restore_motion_snapshot(path, trainer):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "traffic-motion-training-v1":
        raise ValueError("Checkpoint does not contain resumable motion training state")
    expected = {
        "domain": trainer.system.domain,
        "config": asdict(trainer.system.cfg),
        "encoder_config": asdict(trainer.system.encoder_cfg),
        "training": asdict(trainer.options),
        "objective": asdict(trainer.objective),
        "train_digest": trainer.train_digest,
        "validation_digest": trainer.validation_digest,
    }
    for name, value in expected.items():
        if state[name] != value:
            raise ValueError(f"Motion resume changed {name}")
    trainer.system.model.load_state_dict(state["model"], strict=True)
    trainer.system.optimizer.load_state_dict(state["optimizer"])
    trainer.scaler.load_state_dict(state["scaler"])
    trainer.train_loader.sampler.load_state_dict(state["sampler"])
    restore_rng(state["rng"], trainer.train_loader)
    return state["epoch"], state["best"], state["history"]
