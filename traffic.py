"""Traffic video preparation, training, memory construction, and inference."""

import argparse
import json
import numpy as np
import math
import random
from pathlib import Path
from dataclasses import asdict
from trafficrag.models.motion import (
    MotionSystem,
    VideoMAEEncoder,
    VideoSource,
    execution_device,
)
from trafficrag.experiments.recipes import (
    catalog_profiles,
    read_settings,
    load_recipe,
    recipes_cli,
    build_recipes_cli,
)
from trafficrag.models.backends import VideoLanguageSystem, create_backend, load_query
from trafficrag.grounding import (
    KnowledgeBase,
    Interval,
    TrafficRAG,
    load_kb,
    segment_video,
    temporal_iou,
)


def train_main(args):
    if args.list_profiles:
        print('\n'.join(str(path) for path in catalog_profiles()))
        return
    options = asdict(load_recipe(args.profile).motion) if args.profile else read_settings(args.config)
    for name in ('seed', 'epochs', 'batch_size', 'lr'):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    config = MotionSystem.Config(**options)
    encoder = VideoMAEEncoder.Config(
        weights=args.weights or '',
        cache_dir=args.cache_dir or '',
        offline=args.offline,
        frames=args.frames,
        image_size=args.image_size,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    'motion': asdict(config),
                    'domain': args.domain,
                    'encoder': asdict(encoder),
                    'data': args.data,
                    'device': args.device,
                },
                indent=2,
            )
        )
        return
    if not args.data:
        raise ValueError('--data must name an annotated video-segment JSONL manifest.')
    system = MotionSystem(config, args.domain, encoder, args.device)
    history = system.fit(args.data)
    system.save(args.output, history)


def train_cli():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default='configs/train.yaml')
    source.add_argument('--profile', help='Paired Python or JSON motion/grounding recipe.')
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--data', help='JSONL: video, start, end, label, optional domain.')
    parser.add_argument(
        '--weights', help='Local official ViT-S .pth file or local OpenGVLab/VideoMAE2 snapshot.'
    )
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=350)
    parser.add_argument(
        '--domain',
        choices=['red-light', 'blind-spot-left', 'blind-spot-right'],
        default='red-light',
    )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='outputs/motion')
    train_main(parser.parse_args())


def build_kb_main(args):
    device = execution_device(args.device)
    root = Path(args.input).resolve().parent
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty knowledge base input.')
    ids = [str(row['id']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Knowledge base segment IDs must be unique.')
    if args.exclude_ids:
        excluded = set(Path(args.exclude_ids).read_text().splitlines())
        if excluded.intersection(str(row.get('video_id', row['id'])) for row in rows):
            raise ValueError('Knowledge base overlaps excluded motion-training/query video IDs.')
    config = VideoLanguageSystem.Config(
        vlm=args.vlm,
        text_encoder=args.text_encoder,
        cache_dir=args.cache_dir or '',
        offline=args.offline,
    )
    system = VideoLanguageSystem(config, device)
    captions = []
    for row in rows:
        if row.get('domain', args.domain) != args.domain:
            raise ValueError('Build one knowledge base per violation domain.')
        if 'caption' in row:
            caption = row['caption']
        else:
            path = Path(row['video'])
            video = VideoSource(path if path.is_absolute() else root / path)
            interval = Interval(float(row.get('start', 0)), float(row.get('end', video.duration)))
            caption = system.caption(video, interval, args.domain)
        captions.append(caption)
    embeddings = np.concatenate(
        [system.embed(captions[i : i + args.batch_size]) for i in range(0, len(rows), args.batch_size)]
    )
    kb = KnowledgeBase(embeddings, [row['label'] for row in rows], captions, ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # File handle keeps the requested destination exact, including unusual suffixes.
    with output.open('wb') as stream:
        np.savez(
            stream,
            embeddings=kb.embeddings.astype(np.float32),
            labels=kb.labels,
            captions=np.asarray(captions),
            ids=np.asarray(ids),
        )
    output.with_suffix('.models.json').write_text(
        json.dumps(
            {
                'domain': args.domain,
                'models': asdict(config),
                'pooling': 'attention-mask mean',
                'similarity': 'cosine',
            },
            indent=2,
        )
        + '\n'
    )
    print(json.dumps({'entries': len(rows), 'dimension': embeddings.shape[1], 'output': str(output)}))


def build_kb_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        help='JSONL: id, video, start, end, label; optional caption and video_id.',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--domain',
        choices=['red-light', 'blind-spot-left', 'blind-spot-right'],
        default='red-light',
    )
    parser.add_argument('--vlm', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--text-encoder', default='answerdotai/ModernBERT-base')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--exclude_ids', help='Newline-separated motion-training/query video IDs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    build_kb_main(parser.parse_args())


def inference_main(args):
    if args.list_profiles:
        print('\n'.join(str(path) for path in catalog_profiles()))
        return
    config = load_recipe(args.profile).to_dict()['pipeline'] if args.profile else read_settings(args.config)
    pipeline_config = TrafficRAG.Config.from_dict(config)
    language_config = VideoLanguageSystem.Config(
        vlm=args.vlm,
        text_encoder=args.text_encoder,
        cache_dir=args.cache_dir or '',
        offline=args.offline,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    'pipeline': config,
                    'language_models': asdict(language_config),
                    'kb': args.kb,
                    'video': args.video,
                    'motion_checkpoint': args.motion_checkpoint,
                    'domain': args.domain,
                    'device': args.device,
                },
                indent=2,
            )
        )
        return
    device = execution_device(args.device)
    if not args.kb:
        raise ValueError('Inference requires --kb.')
    metadata_path = Path(args.kb).with_suffix('.models.json')
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata['domain'] != args.domain:
            raise ValueError('Knowledge-base domain must match --domain.')
        if args.video and metadata['models']['text_encoder'] != args.text_encoder:
            raise ValueError('Use the same --text-encoder path/ID for KB construction and queries.')
    pipeline = TrafficRAG(load_kb(args.kb), config=pipeline_config)
    if args.video:
        if not args.motion_checkpoint:
            raise ValueError('Video inference requires a domain-trained --motion_checkpoint.')
        video = VideoSource(args.video)
        motion = MotionSystem.restore(args.motion_checkpoint, device)
        if motion.domain != args.domain:
            raise ValueError('The motion checkpoint domain must match --domain.')
        proposal = pipeline_config.proposal
        segments = segment_video(video.duration, proposal.window, proposal.stride)
        scores = motion.score_video(video, segments)
        # Stage 1 finishes before allocating the large vision-language model.
        del motion
        import torch

        torch.cuda.empty_cache()
        backend = VideoLanguageSystem(language_config, device).for_video(video, args.domain)
        duration = video.duration
        mode = 'VideoMAE V2 Small + Qwen3-VL + ModernBERT'
    elif args.query:
        query = load_query(args.query)
        query['device'] = str(device)
        backend = create_backend(args.backend or 'recorded', query)
        duration, scores = query['duration'], query['motion_scores']
        mode = 'recorded model outputs' if not args.backend else 'external callbacks'
    else:
        raise ValueError('Supply --video for model-backed inference or --query for recorded outputs.')
    result = pipeline(duration, scores, backend.caption, backend.embed, backend.ground)
    result.update(execution_mode=mode, domain=args.domain)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + '\n')


def inference_cli():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default='configs/default.yaml')
    source.add_argument('--profile', help='Paired Python or JSON motion/grounding recipe.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--video', help='Raw dashcam video file.')
    inputs.add_argument('--query', help='Recorded-output JSON for inspecting saved model predictions.')
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--kb')
    parser.add_argument('--motion_checkpoint')
    parser.add_argument(
        '--domain',
        choices=['red-light', 'blind-spot-left', 'blind-spot-right'],
        default='red-light',
    )
    parser.add_argument('--vlm', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--text-encoder', default='answerdotai/ModernBERT-base')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--backend', help='Recorded-query override: module:factory.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    inference_main(parser.parse_args())


def eval_main(args):
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No evaluation rows.')
    tp = fp = fn = tn = 0
    ious = []
    for row in rows:
        pred = Interval(*row['prediction']) if row['prediction'] is not None else None
        truth = Interval(*row['target']) if row['target'] is not None else None
        tp += int(pred is not None and truth is not None)
        fp += int(pred is not None and truth is None)
        fn += int(pred is None and truth is not None)
        tn += int(pred is None and truth is None)
        if truth is not None:
            ious.append(temporal_iou(pred, truth))
    print(
        json.dumps(
            {
                'classification_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                'positive_video_mean_iou': float(np.mean(ious)) if ious else None,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            },
            indent=2,
        )
    )


def eval_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        required=True,
        help='JSONL: prediction [start,end] or null, target [start,end] or null.',
    )
    eval_main(parser.parse_args())


def prepare_splits_main(args):
    source = Path(args.annotations).resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if (
        not 0 < args.validation_fraction < 1
        or not 0 < args.kb_fraction < 1
        or args.validation_fraction + args.kb_fraction >= 1
    ):
        raise ValueError('Validation/KB fractions must be positive and leave a training fraction.')
    video_paths, path_ids = {}, {}
    for index, row in enumerate(rows):
        video_id = str(row['video_id'])
        path = Path(row['video'])
        path = path if path.is_absolute() else source.parent / path
        path = str(path.resolve())
        if video_id in video_paths and video_paths[video_id] != path:
            raise ValueError('One video_id must refer to one acquired video file.')
        if path in path_ids and path_ids[path] != video_id:
            raise ValueError('Use one video_id for every segment of the same file.')
        video_paths[video_id], path_ids[path] = path, video_id
        start, end = float(row['start']), float(row['end'])
        if (
            row['label'] not in (0, 1)
            or not all(math.isfinite(v) for v in (start, end))
            or not 0 <= start < end
        ):
            raise ValueError('Each row needs label 0/1 and finite 0 <= start < end.')
        row.update(video=path, id=str(row.get('id', f'{video_id}_{index:06d}')))
    videos = sorted(video_paths)
    if len(videos) < 3:
        raise ValueError('At least three distinct videos are needed for disjoint splits.')
    random.Random(args.seed).shuffle(videos)
    n_val = min(len(videos) - 2, max(1, round(len(videos) * args.validation_fraction)))
    n_kb = min(len(videos) - n_val - 1, max(1, round(len(videos) * args.kb_fraction)))
    validation, knowledge = set(videos[:n_val]), set(videos[n_val : n_val + n_kb])
    splits = {'motion_train': [], 'validation': [], 'kb': []}
    for row in rows:
        name = (
            'validation'
            if str(row['video_id']) in validation
            else 'kb' if str(row['video_id']) in knowledge else 'motion_train'
        )
        splits[name].append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, entries in splits.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entries))
    (output / 'train_query_ids.txt').write_text(''.join(v + '\n' for v in videos if v not in knowledge))
    print(json.dumps({name: len(entries) for name, entries in splits.items()}))


def prepare_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', default='data')
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--kb-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    prepare_splits_main(parser.parse_args())


def main():
    import sys

    from trafficrag.data.records import manifest_cli
    from trafficrag.data.partitions import partitions_cli
    from trafficrag.data.artifacts import cache_cli
    from trafficrag.runner import batch_cli
    from trafficrag.experiments.summary import report_cli

    commands = {
        'manifest': manifest_cli,
        'partitions': partitions_cli,
        'cache': cache_cli,
        'batch': batch_cli,
        'report': report_cli,
        'train': train_cli,
        'build-kb': build_kb_cli,
        'infer': inference_cli,
        'evaluate': eval_cli,
        'prepare': prepare_cli,
        'recipes': recipes_cli,
        'build-recipes': build_recipes_cli,
    }
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print("Commands: " + ", ".join(commands))
        print("Use python traffic.py <command> --help for stage options.")
        return
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"Usage: python traffic.py <command> [options]\nCommands: {choices}")
    command = sys.argv.pop(1)
    commands[command]()


if __name__ == "__main__":
    main()
