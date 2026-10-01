# blind-spot-right workflow

Prepare recordings and interval labels for the `blind-spot-right` event criterion. Keep complete drive groups together when assigning motion-training, query, and caption-memory records.

```bash
python traffic.py partitions --manifest data/blind_right/annotations.jsonl --output data/blind_right/splits --seed 42
python traffic.py train --data data/blind_right/splits/motion_train.jsonl --domain blind-spot-right --output outputs/motion/blind_right --device cuda
python traffic.py build-kb --input data/blind_right/splits/kb.jsonl --exclude_ids data/blind_right/splits/train_query_ids.txt --domain blind-spot-right --output data/blind_right/kb.npz --device cuda
python traffic.py batch --manifest data/blind_right/queries.jsonl --motion-checkpoint outputs/motion/blind_right/last.pt --kb data/blind_right/kb.npz --domain blind-spot-right --output outputs/blind_right --device cuda
python traffic.py report --predictions outputs/blind_right/predictions.jsonl --output reports/blind_right --bootstrap 1000
```

The query manifest identifies the whole recording and its reviewed target interval. Its IDs match the reporting target IDs. Add `--resume` to the batch command to continue the same saved run.
