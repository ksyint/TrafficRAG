"""Keep all segments of each video in one motion/validation/knowledge split."""
import argparse
import json
import math
from pathlib import Path
import random


def main(args):
    source = Path(args.annotations).resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not 0 < args.validation_fraction < 1 or not 0 < args.kb_fraction < 1 or args.validation_fraction + args.kb_fraction >= 1:
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
        if row['label'] not in (0, 1) or not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError('Each row needs label 0/1 and finite 0 <= start < end.')
        row.update(video=path, id=str(row.get('id', f'{video_id}_{index:06d}')))
    videos = sorted(video_paths)
    if len(videos) < 3:
        raise ValueError('At least three distinct videos are needed for disjoint splits.')
    random.Random(args.seed).shuffle(videos)
    n_val = min(len(videos) - 2, max(1, round(len(videos) * args.validation_fraction)))
    n_kb = min(len(videos) - n_val - 1, max(1, round(len(videos) * args.kb_fraction)))
    validation, knowledge = set(videos[:n_val]), set(videos[n_val:n_val + n_kb])
    splits = {'motion_train': [], 'validation': [], 'kb': []}
    for row in rows:
        name = 'validation' if str(row['video_id']) in validation else 'kb' if str(row['video_id']) in knowledge else 'motion_train'
        splits[name].append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, entries in splits.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entries))
    (output / 'train_query_ids.txt').write_text(''.join(v + '\n' for v in videos if v not in knowledge))
    print(json.dumps({name: len(entries) for name, entries in splits.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', default='data')
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--kb-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
