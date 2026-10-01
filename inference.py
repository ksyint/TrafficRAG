import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from trafficrag import TrafficRAG
from trafficrag.backends import create_backend
from trafficrag.backends.manifest import load_query
from trafficrag.experiments import catalog_profiles, load_recipe
from trafficrag.knowledge import load_kb
from trafficrag.motion import MotionSystem
from trafficrag.runtime import execution_device


def main(args):
    if args.list_profiles:
        print('\n'.join(str(path) for path in catalog_profiles()))
        return
    config = load_recipe(args.profile).to_dict()['pipeline'] if args.profile else yaml.safe_load(Path(args.config).read_text())
    pipeline_config = TrafficRAG.Config.from_dict(config)
    if args.dry_run:
        print(json.dumps({'pipeline': config, 'kb': args.kb, 'query': args.query,
                          'motion_checkpoint': args.motion_checkpoint, 'backend': args.backend or 'recorded',
                          'device': args.device}, indent=2))
        return
    device = execution_device(args.device)
    if not args.kb or not args.query:
        raise ValueError('Inference requires --kb and --query.')
    query = load_query(args.query)
    query['device'] = str(device)
    pipeline = TrafficRAG(load_kb(args.kb), config=pipeline_config)
    backend = create_backend(args.backend or 'recorded', query)
    if args.motion_checkpoint:
        data_path = Path(query['motion_data'])
        if not data_path.is_absolute():
            data_path = Path(args.query).parent / data_path
        features = torch.from_numpy(np.load(data_path, allow_pickle=False)).float()
        scores = MotionSystem.score(args.motion_checkpoint, features, device)
    else:
        scores = query['motion_scores']
    result = pipeline(query['duration'], scores, backend.caption, backend.embed, backend.ground)
    result['execution_mode'] = 'external callbacks' if args.backend else 'recorded model outputs'
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
    parser.add_argument('--list-profiles', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--query')
    parser.add_argument('--kb')
    parser.add_argument('--motion_checkpoint')
    parser.add_argument('--backend', help='module:factory; factory(query) returns caption, embed, ground methods.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    main(parser.parse_args())
