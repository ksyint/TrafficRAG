# red-light workflow

Prepare recordings and interval labels for the `red-light` event criterion. Keep complete drive groups together when assigning motion-training, query, and caption-memory records.

```bash
python traffic.py partitions --manifest data/red_light/annotations.jsonl --output data/red_light/splits --seed 42
python traffic.py train --data data/red_light/splits/motion_train.jsonl --domain red-light --output outputs/motion/red_light --device cuda
python traffic.py build-kb --input data/red_light/splits/kb.jsonl --exclude_ids data/red_light/splits/train_query_ids.txt --domain red-light --output data/red_light/kb.npz --device cuda
python traffic.py batch --manifest data/red_light/queries.jsonl --motion-checkpoint outputs/motion/red_light/last.pt --kb data/red_light/kb.npz --domain red-light --output outputs/red_light --device cuda
python traffic.py report --predictions outputs/red_light/predictions.jsonl --output reports/red_light --bootstrap 1000
```

The query manifest identifies the whole recording and its reviewed target interval. Its IDs match the reporting target IDs. Add `--resume` to the batch command to continue the same saved run.
