from dataclasses import asdict, dataclass
import json
from pathlib import Path

from trafficrag.motion import MotionSystem
from trafficrag.systems.grounding import TrafficRAG


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'configs' / 'catalog'


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
        return {'version': self.version, 'pipeline': asdict(self.pipeline), 'motion': asdict(self.motion)}


def load_recipe(path):
    return ExperimentRecipe.from_dict(json.loads(Path(path).read_text()))


def catalog_profiles():
    return sorted(CATALOG.rglob('*.json'))
