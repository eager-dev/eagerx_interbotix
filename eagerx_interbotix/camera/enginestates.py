import eagerx
from eagerx.core.specs import EngineStateSpec
from typing import Any


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        pass

    def reset(self, state: Any):
        pass
