import numpy as np
from typing import List, Any
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec, ObjectSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register

# IMPORT INTERBOTIX
from interbotix_xs_modules import arm, gripper, core


def get_joint_indices(info, joints):
    joint_indices = []
    for n in joints:
        i = info.joint_names.index(n)
        joint_indices.append(info.joint_state_indices[i])
    return joint_indices


class XseriesSensor(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[str],
        process: int = eagerx.NEW_PROCESS,
        color: str = "cyan",
        mode: str = "position",
    ) -> NodeSpec:
        """XseriesSensor spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.joints = joints

        # Set variable spaces
        rng = np.pi * np.ones(len(joints), dtype="float32")
        spec.outputs.obs.space.update(low=-rng, high=rng)
        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        if not spec.config.mode == "position":
            raise NotImplementedError(f"This mode is not implemented: {spec.config.mode}")
        self.joints = spec.config.joints
        self.mode = spec.config.mode
        self.dxl = core.InterbotixRobotXSCore(object_spec.config.robot_type, object_spec.config.name, False)
        self.arm = arm.InterbotixArmXSInterface(
            self.dxl, object_spec.config.robot_type, "arm", moving_time=0.2, accel_time=0.3
        )

        # Determine joint order
        self.joint_indices = get_joint_indices(self.dxl.robot_get_robot_info("group", "arm"), spec.config.joints)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(obs=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        joint_state = self.dxl.robot_get_joint_states()
        obs = [joint_state.position[i] for i in self.joint_indices]
        return dict(obs=np.array(obs, dtype="float32"))

    def close(self):
        pass


class XseriesArm(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[str],
        process: int = eagerx.NEW_PROCESS,
        color: str = "green",
        mode: str = "position_control",
    ) -> NodeSpec:
        """XseriesArm spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.joints = joints

        # Set variable spaces
        rng = np.pi * np.ones(len(joints), dtype="float32")
        spec.inputs.action.space.update(low=-rng, high=rng)
        spec.outputs.action_applied.space.update(low=-rng, high=rng)

        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        if not spec.config.mode == "position_control":
            raise NotImplementedError(f"This mode is not implemented: {spec.config.mode}")
        self.mode = spec.config.mode
        self.dxl = core.InterbotixRobotXSCore(object_spec.config.robot_type, object_spec.config.name, False)
        self.arm = arm.InterbotixArmXSInterface(
            self.dxl, object_spec.config.robot_type, "arm", moving_time=4 / self.rate, accel_time=1 / self.rate
        )

        # Determine joint order
        self.joint_indices = get_joint_indices(self.dxl.robot_get_robot_info("group", "arm"), spec.config.joints)

        self.arm.go_to_home_pose(moving_time=2.0, accel_time=0.3)

    @register.states()
    def reset(self):
        self.last_cmd = None

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg, action: Msg):
        positions = action.msgs[-1]
        indexed_positions = [positions[i] for i in self.joint_indices]
        if not self.last_cmd == indexed_positions:
            self.last_cmd = indexed_positions
            self.arm.set_joint_positions(indexed_positions, moving_time=2.5, accel_time=0.3, blocking=False)
        self.arm.set_joint_positions(indexed_positions, moving_time=4 / self.rate, accel_time=1 / self.rate, blocking=False)
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    def shutdown(self):
        self.dxl.robot_torque_enable("group", "arm", True)


class XseriesGripper(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: int = eagerx.NEW_PROCESS,
        color: str = "green",
    ) -> NodeSpec:
        """XseriesGripper spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]
        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        self.dxl = core.InterbotixRobotXSCore(object_spec.config.robot_type, object_spec.config.name, False)
        self.gripper = gripper.InterbotixGripperXSInterface(
            self.dxl,
            "gripper",
            gripper_pressure=0.5,
            gripper_pressure_lower_limit=150,
            gripper_pressure_upper_limit=350,
        )

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(low=[0], high=[1], dtype="float32"))
    @register.outputs(action_applied=Space(low=[0], high=[1], dtype="float32"))
    def callback(self, t_n: float, tick: Msg, action: Msg):
        # Maps action=[0, 1.] to [-gripper_value, gripper_value].
        gripper_value = self.gripper.gripper_value * ((action.msgs[-1][0] * 2) - 1)
        # Set gripper value
        self.gripper.gripper_controller(gripper_value, delay=0.0)
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    def shutdown(self):
        self.gripper.open(delay=0.0)
