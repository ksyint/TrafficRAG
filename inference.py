import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.data import RecordedBackend, load_kb, load_query
from utils.models import BinaryMotionClassifier, VideoMAEEncoder
from utils.pipeline import KnowledgeBase, TrafficRAG, segment_video


def smoke():
    # Controlled callbacks test plumbing; they do not claim VLM inference.
    kb = KnowledgeBase(np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]), [1, 1, 0, 0])
    pipeline = TrafficRAG(kb, topk=2, smoothing_sigma=0)
    result = pipeline(8, [0.05, 0.1, 0.9, 0.8, 0.1, 0.05, 0.02],
                      lambda interval: 'Vehicle crosses a stop line while the signal is red.',
                      lambda caption: np.array([1.0, 0.0]),
                      lambda crop, candidate: (2.4 - crop.start, 3.6 - crop.start))
    result['execution_mode'] = 'synthetic callbacks; no pretrained models'
    return result


def main(args):
    if args.smoke:
        result = smoke()
    else:
        if not args.kb or not args.query:
            raise ValueError('--kb and --query are required unless --smoke is selected.')
        config = yaml.safe_load(Path(args.config).read_text())
        query = load_query(args.query)
        pipeline = TrafficRAG(load_kb(args.kb), **config)
        if args.backend:
            module, name = args.backend.split(':')
            backend = getattr(importlib.import_module(module), name)(query)
        else:
            backend = RecordedBackend(query)
        if args.motion_checkpoint:
            checkpoint = torch.load(args.motion_checkpoint, map_location=args.device, weights_only=True)
            encoder = VideoMAEEncoder(checkpoint['encoder']) if checkpoint['encoder'] else None
            model = BinaryMotionClassifier(checkpoint['feature_dim'], encoder).to(args.device).eval()
            model.load_state_dict(checkpoint['model'])
            data_path = Path(query['motion_data'])
            if not data_path.is_absolute():
                data_path = Path(args.query).parent / data_path
            x = torch.from_numpy(np.load(data_path, allow_pickle=False)).float()
            with torch.no_grad():
                scores = torch.cat([model(chunk.to(args.device)).sigmoid().cpu() for chunk in x.split(8)]).numpy()
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
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--query')
    parser.add_argument('--kb')
    parser.add_argument('--motion_checkpoint')
    parser.add_argument('--backend', help='module:factory; factory(query) returns caption, embed, ground methods.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output')
    main(parser.parse_args())
