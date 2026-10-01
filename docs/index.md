# Workflow guides

## Overview

- [Runtime packages](architecture.md)

## Data

- [Segment annotations](data/annotations.md)
- [Recording and query identities](data/identities.md)
- [Group-preserving partitions](data/partitions.md)
- [Stage artifact cache](data/stage-cache.md)
- [Video timestamps and frame preparation](data/video-decoding.md)

## Evaluation

- [Recording-level confidence intervals](evaluation/bootstrap.md)
- [Inspecting retrieved evidence](evaluation/evidence.md)
- [Paired grounding comparisons](evaluation/paired.md)
- [Event presence metrics](evaluation/presence.md)
- [Temporal localization metrics](evaluation/temporal.md)
- [Red-light events](evaluation/domain-red_light.md)
- [Left-side blind-spot events](evaluation/domain-blind_left.md)
- [Right-side blind-spot events](evaluation/domain-blind_right.md)

## Models

- [Captioning and temporal generation](grounding/captioning.md)
- [VideoMAE motion lifecycle](grounding/motion.md)
- [Local pretrained artifacts](grounding/offline.md)
- [ModernBERT caption retrieval](grounding/retrieval.md)

## Pipeline

- [Batch grounding and continuation](grounding/batch-resume.md)
- [Motion proposals](grounding/proposals.md)
- [Adaptive boundary refinement](grounding/refinement.md)
- [Semantic verification](grounding/verification.md)
