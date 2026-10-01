import argparse
from pathlib import Path
from dataclasses import asdict
import json

import yaml

from trafficrag.motion import MotionSystem
from trafficrag.motion.encoder import VideoMAEEncoder
from trafficrag.experiments import catalog_profiles, load_recipe


def main(args):
    if args.list_profiles:
        print('\n'.join(str(path) for path in catalog_profiles()))
        return
    options = asdict(load_recipe(args.profile).motion) if args.profile else yaml.safe_load(Path(args.config).read_text())
    for name in ('seed', 'epochs', 'batch_size', 'lr'):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    config = MotionSystem.Config(**options)
    encoder = VideoMAEEncoder.Config(weights=args.weights or '', cache_dir=args.cache_dir or '',
                                     offline=args.offline, frames=args.frames, image_size=args.image_size)
    if args.dry_run:
        print(json.dumps({'motion': asdict(config), 'domain': args.domain, 'encoder': asdict(encoder),
                          'data': args.data, 'device': args.device}, indent=2))
        return
    if not args.data:
        raise ValueError('--data must name an annotated video-segment JSONL manifest.')
    system = MotionSystem(config, args.domain, encoder, args.device)
    history = system.fit(args.data)
    system.save(args.output, history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default='configs/train.yaml')
    source.add_argument('--profile', help='Paired JSON motion/grounding recipe.')
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--data', help='JSONL: video, start, end, label, optional domain.')
    parser.add_argument('--weights', help='Local official ViT-S .pth file or local OpenGVLab/VideoMAE2 snapshot.')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=350)
    parser.add_argument('--domain', choices=['red-light', 'blind-spot-left', 'blind-spot-right'], default='red-light')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='outputs/motion')
    main(parser.parse_args())
