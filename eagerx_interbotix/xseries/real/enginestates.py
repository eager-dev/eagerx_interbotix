from typing import Any
from eagerx.core.specs import EngineStateSpec, ObjectSpec
from interbotix_copilot.client import Client
import eagerx
import numpy as np


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        pass

    def reset(self, state: np.ndarray):
        pass


class CopilotStateReset(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        # Get arm client
        arm_name = object_spec.config.name
        if "client" not in simulator[arm_name]:
            simulator[arm_name]["client"] = Client(object_spec.config.robot_type, arm_name, group_name="arm")
        self.arm: Client = simulator[arm_name]["client"]

    def reset(self, state: np.ndarray):
        f = self.arm.wait_for_feedthrough()
        f.result()  # Wait for feedthrough to be toggled on.
        f = self.arm.go_to(points=list(state), timestamps=5.0)
        f.result()  # Wait for the desired position to be reached.
