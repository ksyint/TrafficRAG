from dataclasses import dataclass


@dataclass
class VerificationConfig:
    semantic_weight: float = 0.4
    topk: int = 10

    def __post_init__(self):
        if not 0 <= self.semantic_weight <= 1 or self.topk < 1:
            raise ValueError('Verification needs 0 <= semantic_weight <= 1 and topk >= 1.')


class SemanticVerification:
    def __init__(self, config, knowledge_base):
        self.cfg = config
        self.knowledge_base = knowledge_base

    def __call__(self, candidates, caption, embed):
        for candidate in candidates:
            candidate.caption = caption(candidate.interval)
            neighbors, similarities, ratio = self.knowledge_base.retrieve(embed(candidate.caption), self.cfg.topk)
            candidate.semantic_score = ratio
            weight = self.cfg.semantic_weight
            candidate.verified_score = weight * ratio + (1 - weight) * candidate.motion_score
            candidate.neighbors = [dict(id=self.knowledge_base.ids[int(i)], similarity=float(similarity),
                                        label=int(self.knowledge_base.labels[i]))
                                   for i, similarity in zip(neighbors, similarities)]
        return max(candidates, key=lambda candidate: candidate.verified_score)
