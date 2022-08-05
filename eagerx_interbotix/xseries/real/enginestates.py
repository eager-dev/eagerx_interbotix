from typing import Any
from eagerx.core.specs import EngineStateSpec, ObjectSpec
from interbotix_copilot.client import Client
import eagerx
from threading import get_ident


class DummyState(eagerx.EngineState):
    @classmethod
    def make(cls):
        return cls.get_specification()

    def initialize(self, spec: EngineStateSpec, object_spec: ObjectSpec, simulator: Any):
        pass

    def reset(self, state: Any):
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

    def reset(self, state: Any):
        # todo: go to arbitrary state.
        f = self.arm.wait_for_feedthrough()
        f.result()  # Wait for feedthrough
        f = self.arm.go_to_home()
        f.result()  # Wait for the home position to be reached.
