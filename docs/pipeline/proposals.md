# Motion proposals

The proposal stage samples overlapping windows, smooths their binary probabilities, and joins consecutive scores above the threshold. Windows include the video tail without extending past the recording duration.

`proposal.window`, `proposal.stride`, `proposal.sigma`, and `proposal.threshold` are validated before inference. The paired recipe catalog sets the proposal threshold while preserving the selected motion-training profile.

```bash
python traffic.py infer --config configs/default.yaml --dry-run
```

Saved batch evidence retains raw probabilities and duration. Each final prediction records smoothed probabilities and candidate intervals for later inspection.
