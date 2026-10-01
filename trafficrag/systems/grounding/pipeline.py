from dataclasses import asdict, dataclass, field

from .proposal import MotionProposal, ProposalConfig
from .refinement import BoundaryRefinement, RefinementConfig
from .verification import SemanticVerification, VerificationConfig


class TrafficRAG:
    """Configurable proposal -> verification -> refinement system."""
    @dataclass
    class Config:
        proposal: ProposalConfig = field(default_factory=ProposalConfig)
        verification: VerificationConfig = field(default_factory=VerificationConfig)
        refinement: RefinementConfig = field(default_factory=RefinementConfig)

        @classmethod
        def from_dict(cls, values):
            values = dict(values)
            nested = {'proposal', 'verification', 'refinement'}
            if set(values) <= nested:
                return cls(ProposalConfig(**values.get('proposal', {})),
                           VerificationConfig(**values.get('verification', {})),
                           RefinementConfig(**values.get('refinement', {})))
            # Load earlier flat configurations and keyword construction transparently.
            mapping = {'motion_threshold': ('proposal', 'threshold'), 'smoothing_sigma': ('proposal', 'sigma'),
                       'window': ('proposal', 'window'), 'stride': ('proposal', 'stride'),
                       'semantic_weight': ('verification', 'semantic_weight'), 'topk': ('verification', 'topk'),
                       'base_padding': ('refinement', 'base_padding'), 'adaptive_padding': ('refinement', 'adaptive_padding')}
            grouped = {key: {} for key in nested}
            for key, value in values.items():
                if key not in mapping:
                    raise ValueError(f'Unknown TrafficRAG setting: {key}')
                group, setting = mapping[key]
                grouped[group][setting] = value
            return cls.from_dict(grouped)

    def __init__(self, knowledge_base, config=None, **settings):
        if config is not None and settings:
            raise ValueError('Provide a Config object or keyword settings, not both.')
        self.cfg = config or self.Config.from_dict(settings)
        self.knowledge_base = knowledge_base
        self.configure()

    def configure(self):
        self.propose = MotionProposal(self.cfg.proposal)
        self.verify = SemanticVerification(self.cfg.verification, self.knowledge_base)
        self.refine = BoundaryRefinement(self.cfg.refinement)

    def __call__(self, duration, motion_scores, caption, embed, ground):
        candidates, smoothed = self.propose(duration, motion_scores)
        result = {'interval': None, 'candidates': [], 'smoothed_scores': smoothed.tolist(), 'crop': None}
        if not candidates:
            return result
        best = self.verify(candidates, caption, embed)
        crop, interval = self.refine(best, duration, ground)
        result.update(interval=asdict(interval) if interval is not None else None,
                      candidates=[asdict(candidate) for candidate in candidates],
                      crop=asdict(crop), selected_score=best.verified_score)
        return result
