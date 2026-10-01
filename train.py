import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from utils.models import BinaryMotionClassifier, VideoMAEEncoder


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(args.seed if args.seed is not None else config['seed'])
    torch.set_num_threads(args.threads)
    encoder = VideoMAEEncoder(args.encoder) if args.encoder else None
    if args.data:
        with np.load(args.data, allow_pickle=False) as data:
            x = torch.from_numpy(data['pixels' if encoder else 'features']).float()
            y = torch.from_numpy(data['labels']).float()
    else:
        if encoder:
            raise ValueError('A pretrained encoder requires real --data pixel tensors.')
        x = torch.randn(128, 16)
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    if y.shape != (len(x),) or not torch.isin(y, torch.tensor([0.0, 1.0])).all():
        raise ValueError('One binary label is required per training segment.')
    dim = encoder.feature_dim if encoder else x.shape[-1]
    model = BinaryMotionClassifier(dim, encoder).to(args.device)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size or config['batch_size'], shuffle=True)
    lr = args.lr if args.lr is not None else (config['lr'] if args.data else 0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    logs = []
    for epoch in range(args.epochs or config['epochs']):
        losses = []
        for batch, labels in loader:
            logits = model(batch.to(args.device))
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        row = {'epoch': epoch, 'binary_cross_entropy': float(np.mean(losses))}
        logs.append(row)
        print(json.dumps(row))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'feature_dim': dim, 'encoder': args.encoder,
                'domain': args.domain, 'config': config}, output / 'last.pt')
    (output / 'metrics.json').write_text(json.dumps(logs, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--data', help='NPZ: features N,D or pixels N,T,C,H,W and labels N.')
    parser.add_argument('--encoder', help='Compatible VideoMAE model ID; enables full encoder fine-tuning.')
    parser.add_argument('--domain', choices=['red-light', 'blind-spot-left', 'blind-spot-right'], default='red-light')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='outputs/motion')
    main(parser.parse_args())
