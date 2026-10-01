# Group-preserving partitions

The partition planner assigns complete recording groups to motion training, query evaluation, or caption memory. Larger groups are placed first, with a seeded digest breaking ties. Remaining assignments follow the largest relative capacity deficit.

```bash
python traffic.py partitions --manifest data/annotations.jsonl --output data/red_light --fractions 0.70 0.15 0.15 --seed 42
python traffic.py partitions --train data/red_light/motion_train.jsonl --query data/red_light/validation.jsonl --kb data/red_light/kb.jsonl
```

Outputs include all three JSONL files, `train_query_ids.txt`, and `partitions.json`. The latter records the source digest, group assignment, seed, and per-partition label counts. Review class counts in each domain before training.
