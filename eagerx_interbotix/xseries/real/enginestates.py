from eagerx.core.entities import EngineState
import eagerx.core.register as register


class DummyState(EngineState):
    @staticmethod
    @register.spec("DummyState", EngineState)
    def spec(spec):
        pass

    def initialize(self):
        pass

    def reset(self, state, done):
        pass
