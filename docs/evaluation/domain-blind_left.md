# blind-spot-left workflow

Prepare recordings and interval labels for the `blind-spot-left` event criterion. Keep complete drive groups together when assigning motion-training, query, and caption-memory records.

```bash
python traffic.py partitions --manifest data/blind_left/annotations.jsonl --output data/blind_left/splits --seed 42
python traffic.py train --data data/blind_left/splits/motion_train.jsonl --domain blind-spot-left --output outputs/motion/blind_left --device cuda
python traffic.py build-kb --input data/blind_left/splits/kb.jsonl --exclude_ids data/blind_left/splits/train_query_ids.txt --domain blind-spot-left --output data/blind_left/kb.npz --device cuda
python traffic.py batch --manifest data/blind_left/queries.jsonl --motion-checkpoint outputs/motion/blind_left/last.pt --kb data/blind_left/kb.npz --domain blind-spot-left --output outputs/blind_left --device cuda
python traffic.py report --predictions outputs/blind_left/predictions.jsonl --output reports/blind_left --bootstrap 1000
```

The query manifest identifies the whole recording and its reviewed target interval. Its IDs match the reporting target IDs. Add `--resume` to the batch command to continue the same saved run.
