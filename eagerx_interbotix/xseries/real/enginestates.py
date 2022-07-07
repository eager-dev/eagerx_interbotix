from typing import Any
from eagerx.core.specs import EngineStateSpec, ObjectSpec
import eagerx


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        pass

    def reset(self, state: Any):
        pass
