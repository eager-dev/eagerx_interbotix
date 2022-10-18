from typing import Any
from eagerx.core.specs import EngineStateSpec
import eagerx
from threading import get_ident


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        pass

    def reset(self, state: Any):
        pass


class GoalState(eagerx.EngineState):
    @classmethod
    def make(cls, mode: str):
        """Prepare goal state for GoalObservationSensor to produce observations with.

        :param mode: Type of observation (e.g. `position`, `orientation`).
        :return:
        """
        spec = cls.get_specification()
        spec.config.mode = mode
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.mode = spec.config.mode
        self.simulator = simulator

    def reset(self, state: Any):
        # Set state in simulator object
        self.simulator[self.mode] = state


class HumanReset(eagerx.EngineState):
    @classmethod
    def make(cls, description: str):
        """Prompts the user to reset a state.

        Blocks until the user key presses `enter` to signal the completion of the state reset.

        :param description: The instructions for the user to perform the state reset.
                            Will have the form f"[object_name] {description}: {state}"
        :return: Parameter specification of the state.
        """
        spec = cls.get_specification()
        spec.config.description = description
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.object_name = simulator["name"]
        self.description = spec.config.description

    def reset(self, state: Any):
        input(f"[{get_ident()}][{self.object_name}] {self.description}: {state}")
        print(f"[{get_ident()}][{self.object_name}]: Continuing...")
