# Adaptive boundary refinement

The selected proposal expands by `base_padding + adaptive_padding * (1 - motion_score)` on each side. Expansion is clipped to the video boundaries.

Qwen inspects this crop with the candidate caption and reviewed retrieved examples. A no-violation response produces a null interval. Numerical responses must satisfy finite crop-relative start and end times inside the selected crop.

The final record includes both the crop and refined source-video interval. `trafficrag/pipeline/temporal/grounding.py` owns the conversion and validation. Batch result JSON is described by [batch-result.json](../../schemas/batch-result.json).
