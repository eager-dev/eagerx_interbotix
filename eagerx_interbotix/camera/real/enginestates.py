from eagerx import EngineState
import eagerx.core.register as register


class DummyState(EngineState):
    @staticmethod
    @register.spec("Dummy", EngineState)
    def spec(spec):
        spec.initialize(DummyState)

    def initialize(self):
        pass

    def reset(self, state, done):
        pass