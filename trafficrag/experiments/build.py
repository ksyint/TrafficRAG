"""Build paired motion-training and temporal-grounding configurations."""
from itertools import product
import json

from .recipe import CATALOG, ExperimentRecipe


RETRIEVAL = (1, 3, 5, 10, 20)
FUSION = {'f02': 0.2, 'f04': 0.4, 'f06': 0.6}
THRESHOLDS = {'m02': 0.2, 'm03': 0.3, 'm04': 0.4, 'm05': 0.5}
PADDING = {'p025': 0.25, 'p050': 0.5}
LEARNING_RATES = {'lr1e6': 1e-6, 'lr5e6': 5e-6}


def main():
    count = 0
    for topk, fusion, threshold, padding, learning_rate in product(RETRIEVAL, FUSION, THRESHOLDS, PADDING, LEARNING_RATES):
        values = {
            'version': 1,
            'pipeline': {
                'proposal': {'threshold': THRESHOLDS[threshold], 'sigma': 1.0, 'window': 2.0, 'stride': 1.0},
                'verification': {'semantic_weight': FUSION[fusion], 'topk': topk},
                'refinement': {'base_padding': PADDING[padding], 'adaptive_padding': 1.0},
            },
            'motion': {'seed': 42, 'epochs': 50, 'batch_size': 1,
                       'lr': LEARNING_RATES[learning_rate], 'weight_decay': 0.01},
        }
        recipe = ExperimentRecipe.from_dict(values)
        path = CATALOG / 'traffic' / 'retrieval' / f'k{topk:02d}' / fusion / threshold / padding / (learning_rate + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe.to_dict(), indent=2) + '\n')
        count += 1
    print(f'Built {count} paired motion/grounding recipes.')


if __name__ == '__main__':
    main()
