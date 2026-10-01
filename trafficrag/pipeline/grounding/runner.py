"""Two-pass CUDA grounding with resumable evidence for each manifest record."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from trafficrag.pipeline.grounding.data.artifacts import ArtifactStore, CachedVideoBackend, digest, file_stamp
from trafficrag.pipeline.grounding.data.records import DOMAINS, VideoManifest, finite_interval
from trafficrag.pipeline.grounding.motion import MotionSystem, VideoSource, execution_device
from trafficrag.pipeline.grounding.backends import ModernBERTEncoder, VideoLanguageSystem
from trafficrag.pipeline.grounding.grounding import TrafficRAG, load_kb, segment_video


class BatchGrounder:
    def __init__(
        self,
        manifest,
        motion_checkpoint,
        knowledge_base,
        output,
        pipeline_config,
        language_config,
        device='cuda',
        cache=None,
        resume=False,
    ):
        self.manifest = manifest
        self.motion_checkpoint = Path(motion_checkpoint).resolve()
        self.knowledge_base = Path(knowledge_base).resolve()
        self.output = Path(output).resolve()
        self.config = pipeline_config
        self.language_config = language_config
        self.device = execution_device(device)
        self.cache = Path(cache or self.output / 'cache').resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        domains = {row.domain for row in manifest}
        if len(domains) != 1:
            raise ValueError('A batch must contain exactly one traffic domain')
        self.domain = domains.pop()
        metadata_path = self.knowledge_base.with_suffix('.models.json')
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata['domain'] != self.domain:
                raise ValueError('Knowledge base belongs to a different traffic domain')
            if metadata['models']['text_encoder'] != language_config.text_encoder:
                raise ValueError('Query and knowledge-base text encoders must match')
        self.signature = {
            'format': 'traffic-batch-v1',
            'manifest': manifest.digest(include_sources=True),
            'motion': file_stamp(self.motion_checkpoint, content=True),
            'kb': file_stamp(self.knowledge_base, content=True),
            'pipeline': asdict(self.config),
            'language': asdict(self.language_config),
            'domain': self.domain,
        }
        identity_path = self.output / 'run.json'
        if identity_path.exists():
            existing = json.loads(identity_path.read_text())
            if not resume:
                raise FileExistsError(
                    'Output already contains a run. Use --resume or another directory'
                )
            if existing != self.signature:
                raise ValueError('Resume inputs, model identity, or pipeline configuration changed')
        else:
            identity_path.write_text(json.dumps(self.signature, indent=2) + '\n')
        self.motion_cache = ArtifactStore(self.cache, 'motion')
        self.records_dir = self.output / 'records'
        self.records_dir.mkdir(exist_ok=True)
        self.completed = self._completed()

    def _result_path(self, identifier):
        return self.records_dir / (digest({'id': identifier}) + '.json')

    def _completed(self):
        results = {}
        for path in sorted(self.records_dir.glob('*.json')):
            row = json.loads(path.read_text())
            identifier = row['id']
            if identifier not in self.manifest.by_id:
                raise ValueError(f'Output contains an unknown record ID: {identifier}')
            if row.get('run_digest') != digest(self.signature):
                raise ValueError(f'Output belongs to a different run: {path}')
            results[identifier] = row
        return results

    def _motion_key(self, record):
        return {
            'source': record.source_stamp(),
            'checkpoint': self.signature['motion'],
            'proposal': asdict(self.config.proposal),
            'domain': self.domain,
        }

    def motion_pass(self, pending):
        model = None
        for record in pending:
            key = self._motion_key(record)
            saved = self.motion_cache.read_json(key)
            if saved is not None:
                finite_interval(record.target, saved['duration'])
                continue
            if model is None:
                model = MotionSystem.restore(self.motion_checkpoint, self.device)
                if model.domain != self.domain:
                    raise ValueError('Motion checkpoint domain differs from the batch')
            video = VideoSource(record.video)
            finite_interval(record.target, video.duration)
            proposal = self.config.proposal
            segments = segment_video(video.duration, proposal.window, proposal.stride)
            scores = model.score_video(video, segments)
            self.motion_cache.write_json(key, {'duration': video.duration, 'scores': scores})
        if model is not None:
            del model
            torch.cuda.empty_cache()

    def _language_system(self):
        system = VideoLanguageSystem(self.language_config, self.device)
        system.configure()
        system.text_encoder = ModernBERTEncoder(
            self.language_config.text_encoder,
            self.device,
            cache_dir=self.language_config.cache_dir or None,
            offline=self.language_config.offline,
            revision=self.language_config.revision,
        )
        identity = dict(asdict(self.language_config))
        identity['resolved_vlm'] = getattr(system.model.config, '_commit_hash', None)
        identity['resolved_text'] = getattr(system.text_encoder.model.config, '_commit_hash', None)
        for key in ('vlm', 'text_encoder'):
            local = Path(identity[key])
            if local.is_dir():
                identity[key + '_files'] = [
                    file_stamp(p)
                    for p in sorted(local.rglob('*'))
                    if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin')
                ]
        return system, identity

    def _store(self, record, prediction, elapsed, cache_stats):
        result = dict(prediction)
        result.update(
            id=record.identifier,
            video_id=record.video_id,
            group=record.group,
            domain=record.domain,
            target=record.target,
            elapsed_seconds=elapsed,
            run_digest=digest(self.signature),
            cache=cache_stats,
        )
        result['prediction'] = result.get('interval')
        payload = json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False).encode()
        ArtifactStore._atomic(self._result_path(record.identifier), payload)
        self.completed[record.identifier] = result
        return result

    def run(self):
        pending = [record for record in self.manifest if record.identifier not in self.completed]
        if pending:
            self.motion_pass(pending)
            pipeline = TrafficRAG(load_kb(self.knowledge_base), config=self.config)
            system = identity = None
            for record in pending:
                start = time.perf_counter()
                saved = self.motion_cache.read_json(self._motion_key(record))
                candidates, _ = pipeline.propose(saved['duration'], saved['scores'])
                if candidates and system is None:
                    system, identity = self._language_system()
                if candidates:
                    video = VideoSource(record.video)
                    backend = CachedVideoBackend(
                        system.for_video(video, self.domain), self.cache, record.video, identity
                    )
                    result = pipeline(
                        saved['duration'],
                        saved['scores'],
                        backend.caption,
                        backend.embed,
                        backend.ground,
                    )
                    stats = backend.summary()
                else:
                    result = {
                        'interval': None,
                        'candidates': [],
                        'crop': None,
                        'smoothed_scores': pipeline.propose(saved['duration'], saved['scores'])[
                            1
                        ].tolist(),
                    }
                    stats = []
                self._store(record, result, time.perf_counter() - start, stats)
        ordered = [self.completed[record.identifier] for record in self.manifest]
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in ordered)
        ArtifactStore._atomic(self.output / 'predictions.jsonl', payload.encode())
        summary = {
            'completed': len(ordered),
            'domain': self.domain,
            'positive_predictions': sum(row['prediction'] is not None for row in ordered),
            'motion_cache': self.motion_cache.summary(),
        }
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary


def batch_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--motion-checkpoint', required=True)
    parser.add_argument('--kb', required=True)
    parser.add_argument('--domain', choices=DOMAINS, required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-cache')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--vlm', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--text-encoder', default='answerdotai/ModernBERT-base')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    manifest = VideoManifest.read(args.manifest, args.domain, require_target=True)
    pipeline = TrafficRAG.Config.from_dict(yaml.safe_load(Path(args.config).read_text()))
    language = VideoLanguageSystem.Config(
        vlm=args.vlm,
        text_encoder=args.text_encoder,
        revision=args.revision,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    runner = BatchGrounder(
        manifest,
        args.motion_checkpoint,
        args.kb,
        args.output,
        pipeline,
        language,
        args.device,
        args.artifact_cache,
        args.resume,
    )
    print(json.dumps(runner.run(), indent=2))
