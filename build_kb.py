import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from trafficrag.backends.pretrained import VideoLanguageSystem
from trafficrag.knowledge import KnowledgeBase
from trafficrag.media import VideoSource
from trafficrag.runtime import execution_device
from trafficrag.systems.grounding.types import Interval


def main(args):
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
    config = VideoLanguageSystem.Config(vlm=args.vlm, text_encoder=args.text_encoder,
                                        cache_dir=args.cache_dir or '', offline=args.offline)
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
    embeddings = np.concatenate([system.embed(captions[i:i + args.batch_size])
                                 for i in range(0, len(rows), args.batch_size)])
    kb = KnowledgeBase(embeddings, [row['label'] for row in rows], captions, ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # File handle keeps the requested destination exact, including unusual suffixes.
    with output.open('wb') as stream:
        np.savez(stream, embeddings=kb.embeddings.astype(np.float32), labels=kb.labels,
                 captions=np.asarray(captions), ids=np.asarray(ids))
    output.with_suffix('.models.json').write_text(json.dumps({'domain': args.domain,
        'models': asdict(config), 'pooling': 'attention-mask mean', 'similarity': 'cosine'}, indent=2) + '\n')
    print(json.dumps({'entries': len(rows), 'dimension': embeddings.shape[1], 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL: id, video, start, end, label; optional caption and video_id.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--domain', choices=['red-light', 'blind-spot-left', 'blind-spot-right'], default='red-light')
    parser.add_argument('--vlm', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--text-encoder', default='answerdotai/ModernBERT-base')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--exclude_ids', help='Newline-separated motion-training/query video IDs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
