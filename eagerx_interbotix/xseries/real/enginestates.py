from typing import Any
from eagerx.core.specs import EngineStateSpec
from interbotix_copilot.client import Client
import eagerx
import numpy as np


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        pass

    def reset(self, state: np.ndarray):
        pass


class CopilotStateReset(eagerx.EngineState):
    @classmethod
    def make(cls, arm_name: str, robot_type: str):
        spec = cls.get_specification()
        spec.config.update(arm_name=arm_name, robot_type=robot_type)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        # Get arm client
        if "client" not in simulator:
            simulator["client"] = Client(spec.config.robot_type, spec.config.arm_name, group_name="arm")
        self.arm: Client = simulator["client"]

    def reset(self, state: np.ndarray):
        f = self.arm.wait_for_feedthrough()
        f.result()  # Wait for feedthrough to be toggled on.
        f = self.arm.write_commands([0.0 for _ in state])
        # f = self.arm.go_to(points=list(state), timestamps=5.0)
        f.result()  # Wait for the desired position to be reached.
