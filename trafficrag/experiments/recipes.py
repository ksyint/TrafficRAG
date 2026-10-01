"""Paired motion and grounding recipes."""

import json
import argparse
import ast
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from trafficrag.models.motion import MotionSystem
from trafficrag.grounding import TrafficRAG
from itertools import product
from pprint import pformat

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'configs' / 'retrieval'


@dataclass
class ExperimentRecipe:
    pipeline: TrafficRAG.Config
    motion: MotionSystem.Config
    version: int = 1

    @classmethod
    def from_dict(cls, values):
        if set(values) != {'version', 'pipeline', 'motion'} or values['version'] != 1:
            raise ValueError('A version-1 recipe contains exactly version, pipeline, and motion sections.')
        return cls(TrafficRAG.Config.from_dict(values['pipeline']), MotionSystem.Config(**values['motion']))

    def to_dict(self):
        return {
            'version': self.version,
            'pipeline': asdict(self.pipeline),
            'motion': asdict(self.motion),
        }


def read_settings(path):
    path = Path(path)
    source = path.read_text()
    if path.suffix == '.py':
        statements = ast.parse(source, filename=str(path)).body
        if (
            len(statements) != 1
            or not isinstance(statements[0], ast.Assign)
            or len(statements[0].targets) != 1
            or not isinstance(statements[0].targets[0], ast.Name)
            or statements[0].targets[0].id != 'RECIPE'
        ):
            raise ValueError('Python settings require one RECIPE dictionary assignment.')
        values = ast.literal_eval(statements[0].value)
    elif path.suffix == '.json':
        values = json.loads(source)
    elif path.suffix in ('.yaml', '.yml'):
        values = yaml.safe_load(source)
    else:
        raise ValueError(f'Unsupported settings format: {path.suffix}')
    if not isinstance(values, dict):
        raise ValueError('Settings must contain a dictionary.')
    return values


def load_recipe(path):
    return ExperimentRecipe.from_dict(read_settings(path))


def catalog_profiles():
    paths = sorted(
        (path for path in CATALOG.rglob('*') if path.is_file() and path.suffix in ('.json', '.py')),
        key=profile_identity,
    )
    keys = [profile_identity(path) for path in paths]
    if len(keys) != len(set(keys)):
        raise ValueError('A recipe may have only one JSON or Python file.')
    return paths


def profile_identity(path):
    path = Path(path)
    parts = path.stem.split('-')
    if parts[0].startswith('k'):
        topk, fusion, threshold, padding, learning_rate = parts
    else:
        topk = next(parent.name for parent in path.parents if parent.name.startswith('k') and parent.name[1:].isdigit())
        fusion, threshold, padding, learning_rate = parts
    return f'{topk}/{fusion}/{threshold}-{padding}-{learning_rate}'


def recipe_path(topk, fusion, threshold, padding, learning_rate):
    name = f'{fusion}-{threshold}-{padding}-{learning_rate}'
    if (topk, fusion, threshold, padding) == (10, 'f04', 'm03', 'p050'):
        return CATALOG / f'k{topk:02d}-{name}'
    folder = CATALOG / f'k{topk:02d}'
    if (fusion, threshold, padding) == ('f04', 'm03', 'p025'):
        return folder / name
    folder /= fusion
    if topk == 10 and fusion == 'f04' and (threshold, padding) != ('m02', 'p050'):
        folder /= threshold
    return folder / name


RETRIEVAL = (1, 3, 5, 10, 20)
FUSION = {'f02': 0.2, 'f04': 0.4, 'f06': 0.6}
THRESHOLDS = {'m02': 0.2, 'm03': 0.3, 'm04': 0.4, 'm05': 0.5}
PADDING = {'p025': 0.25, 'p050': 0.5}
LEARNING_RATES = {'lr1e6': 1e-6, 'lr5e6': 5e-6}


def build_main():
    count = 0
    coordinates = list(product(RETRIEVAL, FUSION, THRESHOLDS, PADDING, LEARNING_RATES))
    coordinates.sort(key=lambda row: f'k{row[0]:02d}/{row[1]}/{row[2]}-{row[3]}-{row[4]}')
    for index, (topk, fusion, threshold, padding, learning_rate) in enumerate(coordinates):
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
        path = recipe_path(topk, fusion, threshold, padding, learning_rate)
        path = path.with_suffix('.py' if index < 108 else '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        values = recipe.to_dict()
        source = (
            'RECIPE = ' + pformat(values, width=100, sort_dicts=False)
            if path.suffix == '.py'
            else json.dumps(values, indent=2)
        )
        path.write_text(source + '\n')
        path.with_suffix('.json' if path.suffix == '.py' else '.py').unlink(missing_ok=True)
        count += 1
    print(f'Built {count} paired motion/grounding recipes.')


def build_recipes_cli():
    build_main()


def catalog_main():
    parser = argparse.ArgumentParser(description='Inspect grounding/motion recipes without loading encoders.')
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
