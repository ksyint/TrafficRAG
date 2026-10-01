import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from trafficrag import TrafficRAG
from trafficrag.backends import create_backend
from trafficrag.backends.manifest import load_query
from trafficrag.backends.pretrained import VideoLanguageSystem
from trafficrag.experiments import catalog_profiles, load_recipe
from trafficrag.knowledge import load_kb
from trafficrag.media import VideoSource
from trafficrag.motion import MotionSystem
from trafficrag.runtime import execution_device
from trafficrag.systems.grounding.proposal import segment_video


def main(args):
    if args.list_profiles:
        print('\n'.join(str(path) for path in catalog_profiles()))
        return
    config = load_recipe(args.profile).to_dict()['pipeline'] if args.profile else yaml.safe_load(Path(args.config).read_text())
    pipeline_config = TrafficRAG.Config.from_dict(config)
    language_config = VideoLanguageSystem.Config(vlm=args.vlm, text_encoder=args.text_encoder,
                                                cache_dir=args.cache_dir or '', offline=args.offline)
    if args.dry_run:
        print(json.dumps({'pipeline': config, 'language_models': asdict(language_config), 'kb': args.kb,
                          'video': args.video, 'motion_checkpoint': args.motion_checkpoint,
                          'domain': args.domain, 'device': args.device}, indent=2))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default='configs/default.yaml')
    source.add_argument('--profile', help='Paired JSON motion/grounding recipe.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--video', help='Raw dashcam video file.')
    inputs.add_argument('--query', help='Recorded-output JSON for inspecting saved model predictions.')
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--kb')
    parser.add_argument('--motion_checkpoint')
    parser.add_argument('--domain', choices=['red-light', 'blind-spot-left', 'blind-spot-right'], default='red-light')
    parser.add_argument('--vlm', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--text-encoder', default='answerdotai/ModernBERT-base')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--backend', help='Recorded-query override: module:factory.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    main(parser.parse_args())
