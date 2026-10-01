import argparse
import json
from pathlib import Path

import numpy as np

from utils.models import ModernBERTEncoder
from utils.pipeline import KnowledgeBase


def main(args):
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty knowledge base input.')
    captions = [row['caption'] for row in rows]
    ids = [str(row['id']) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Knowledge base IDs must be unique.')
    if args.exclude_ids:
        excluded = set(Path(args.exclude_ids).read_text().splitlines())
        if excluded.intersection(ids):
            raise ValueError('Knowledge base overlaps the excluded training/query video IDs.')
    if args.encoder:
        encoder = ModernBERTEncoder(args.encoder, args.device)
        embeddings = np.concatenate([encoder(captions[i:i + args.batch_size]) for i in range(0, len(rows), args.batch_size)])
    else:
        embeddings = np.asarray([row['embedding'] for row in rows], dtype=np.float32)
    kb = KnowledgeBase(embeddings, [row['label'] for row in rows], captions, ids)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embeddings=kb.embeddings.astype(np.float32), labels=kb.labels,
             captions=np.asarray(captions), ids=np.asarray(ids))
    print(json.dumps({'entries': len(rows), 'dimension': embeddings.shape[1], 'output': args.output}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL: id, caption, label, embedding (unless --encoder).')
    parser.add_argument('--output', required=True)
    parser.add_argument('--encoder', help='ModernBERT model identifier or local path; frozen masked-mean pooling.')
    parser.add_argument('--exclude_ids', help='Optional newline-separated train/query video IDs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    main(parser.parse_args())
