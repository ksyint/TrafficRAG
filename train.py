import argparse
from pathlib import Path
from dataclasses import asdict
import json

import torch
import yaml

from trafficrag.motion import MotionSystem
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
    if not args.data and not args.profile and args.lr is None:
        options['lr'] = 0.03
    config = MotionSystem.Config(**options)
    if args.dry_run:
        print(json.dumps({'motion': asdict(config), 'domain': args.domain, 'encoder': args.encoder,
                          'data': args.data, 'device': args.device}, indent=2))
        return
    if args.profile and not args.data:
        raise ValueError('A motion recipe requires --data with annotated segment features or pixels.')
    torch.set_num_threads(args.threads)
    system = MotionSystem(config, args.domain, args.encoder, args.device)
    history = system.fit(args.data)
    system.save(args.output, history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', default='configs/train.yaml')
    source.add_argument('--profile', help='Paired JSON motion/grounding recipe.')
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--data', help='NPZ: features N,D or pixels N,T,C,H,W and labels N.')
    parser.add_argument('--encoder', help='Compatible VideoMAE model ID; enables full encoder fine-tuning.')
    parser.add_argument('--domain', choices=['red-light', 'blind-spot-left', 'blind-spot-right'], default='red-light')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='outputs/motion')
    main(parser.parse_args())
