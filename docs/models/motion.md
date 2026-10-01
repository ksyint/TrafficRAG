# VideoMAE motion lifecycle

The motion model starts from the released Kinetics-710 distilled VideoMAE V2 Small weights. A domain-specific binary head is trained together with the encoder.

```bash
python traffic.py train --data data/red_light/motion_train.jsonl --domain red-light --output outputs/motion/red_light --device cuda
```

`last.pt` contains the trained encoder, binary head, domain, and configuration. Batch grounding restores that checkpoint once for its motion pass, scores all uncached videos, then releases it before allocating the vision-language model. The pretrained action-recognition initialization is downloaded automatically by the ordinary training command.
