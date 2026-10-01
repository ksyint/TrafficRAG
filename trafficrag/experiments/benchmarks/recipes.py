"""Paired motion and grounding recipes."""

import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from trafficrag.pipeline.grounding.motion import MotionSystem
from trafficrag.pipeline.grounding.grounding import TrafficRAG
from itertools import product

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / 'configs' / 'catalog'


@dataclass
class ExperimentRecipe:
    pipeline: TrafficRAG.Config
    motion: MotionSystem.Config
    version: int = 1

    @classmethod
    def from_dict(cls, values):
        if set(values) != {'version', 'pipeline', 'motion'} or values['version'] != 1:
            raise ValueError(
                'A version-1 recipe contains exactly version, pipeline, and motion sections.'
            )
        return cls(
            TrafficRAG.Config.from_dict(values['pipeline']), MotionSystem.Config(**values['motion'])
        )

    def to_dict(self):
        return {
            'version': self.version,
            'pipeline': asdict(self.pipeline),
            'motion': asdict(self.motion),
        }


def load_recipe(path):
    return ExperimentRecipe.from_dict(json.loads(Path(path).read_text()))


def catalog_profiles():
    return sorted(CATALOG.rglob('*.json'))


RETRIEVAL = (1, 3, 5, 10, 20)
FUSION = {'f02': 0.2, 'f04': 0.4, 'f06': 0.6}
THRESHOLDS = {'m02': 0.2, 'm03': 0.3, 'm04': 0.4, 'm05': 0.5}
PADDING = {'p025': 0.25, 'p050': 0.5}
LEARNING_RATES = {'lr1e6': 1e-6, 'lr5e6': 5e-6}


def build_main():
    count = 0
    for topk, fusion, threshold, padding, learning_rate in product(
        RETRIEVAL, FUSION, THRESHOLDS, PADDING, LEARNING_RATES
    ):
        values = {
            'version': 1,
            'pipeline': {
                'proposal': {
                    'threshold': THRESHOLDS[threshold],
                    'sigma': 1.0,
                    'window': 2.0,
                    'stride': 1.0,
                },
                'verification': {'semantic_weight': FUSION[fusion], 'topk': topk},
                'refinement': {'base_padding': PADDING[padding], 'adaptive_padding': 1.0},
            },
            'motion': {
                'seed': 42,
                'epochs': 50,
                'batch_size': 1,
                'lr': LEARNING_RATES[learning_rate],
                'weight_decay': 0.01,
            },
        }
        recipe = ExperimentRecipe.from_dict(values)
        path = (
            CATALOG
            / 'retrieval'
            / f'k{topk:02d}'
            / fusion
            / f'{threshold}-{padding}-{learning_rate}.json'
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe.to_dict(), indent=2) + '\n')
        count += 1
    print(f'Built {count} paired motion/grounding recipes.')


def build_recipes_cli():
    build_main()


def catalog_main():
    parser = argparse.ArgumentParser(
        description='Inspect grounding/motion recipes without loading encoders.'
    )
    parser.add_argument('--profile')
    parser.add_argument('--validate-all', action='store_true')
    args = parser.parse_args()
    if args.profile:
        print(json.dumps(load_recipe(args.profile).to_dict(), indent=2))
    elif args.validate_all:
        paths = catalog_profiles()
        for path in paths:
            load_recipe(path)
        print(json.dumps({'validated_recipes': len(paths)}))
    else:
        print('\n'.join(path.relative_to(ROOT).as_posix() for path in catalog_profiles()))


def recipes_cli():
    catalog_main()
