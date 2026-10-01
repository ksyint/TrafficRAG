import argparse
import json
from pathlib import Path

import numpy as np

from trafficrag.systems.grounding import Interval
from trafficrag.metrics import temporal_iou


def main(args):
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
    print(json.dumps({'classification_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                      'positive_video_mean_iou': float(np.mean(ious)) if ious else None,
                      'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='JSONL: prediction [start,end] or null, target [start,end] or null.')
    main(parser.parse_args())
