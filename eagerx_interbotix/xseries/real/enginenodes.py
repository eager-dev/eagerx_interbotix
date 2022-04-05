from typing import Optional, List

# IMPORT ROS
from std_msgs.msg import UInt64, Float32MultiArray

# IMPORT EAGERX
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register

# IMPORT INTERBOTIX
from interbotix_xs_modules import arm, gripper, core


def get_joint_indices(info, joints):
    joint_indices = []
    for n in joints:
        i = info.joint_names.index(n)
        joint_indices.append(info.joint_state_indices[i])
    return joint_indices


class XseriesSensor(EngineNode):
    @staticmethod
    @register.spec("XseriesSensor", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "cyan",
        mode: str = "position",
    ):
        """XseriesSensor spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(XseriesSensor)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.joints = joints

    def initialize(self, joints, mode):
        if not mode == "position":
            # todo: implement velocity mode
            raise NotImplementedError(f"This mode is not implemented: {mode}")
        self.joints = joints
        self.mode = mode
        self.dxl = core.InterbotixRobotXSCore(self.config["robot_type"], self.config["name"], False)
        self.arm = arm.InterbotixArmXSInterface(self.dxl, self.config["robot_type"], "arm", moving_time=0.2, accel_time=0.3)

        # Determine joint order
        self.joint_indices = get_joint_indices(self.dxl.robot_get_robot_info("group", "arm"), joints)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(obs=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        joint_state = self.dxl.robot_get_joint_states()
        obs = [joint_state.position[i] for i in self.joint_indices]
        return dict(obs=Float32MultiArray(data=obs))

    def close(self):
        pass


class XseriesArm(EngineNode):
    @staticmethod
    @register.spec("XseriesArm", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "green",
        mode: str = "position_control",
    ):
        """XseriesArm spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(XseriesArm)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.joints = joints

    def initialize(self, joints, mode):
        if not mode == "position_control":
            raise NotImplementedError(f"This mode is not implemented: {mode}")
        self.mode = mode
        self.dxl = core.InterbotixRobotXSCore(self.config["robot_type"], self.config["name"], False)
        self.arm = arm.InterbotixArmXSInterface(
            self.dxl, self.config["robot_type"], "arm", moving_time=4 / self.rate, accel_time=1 / self.rate
        )

        # Determine joint order
        self.joint_indices = get_joint_indices(self.dxl.robot_get_robot_info("group", "arm"), joints)

        self.arm.go_to_home_pose(moving_time=2.0, accel_time=0.3)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64, action=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Msg] = None,
    ):
        positions = action.msgs[-1].data
        indexed_positions = [positions[i] for i in self.joint_indices]
        self.arm.set_joint_positions(indexed_positions, moving_time=4 / self.rate, accel_time=1 / self.rate, blocking=False)
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    def shutdown(self):
        self.dxl.robot_torque_enable("group", "arm", True)


class XseriesGripper(EngineNode):
    @staticmethod
    @register.spec("XseriesGripper", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        process: Optional[int] = process.NEW_PROCESS,
        color: Optional[str] = "green",
    ):
        """XseriesGripper spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(XseriesGripper)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

    def initialize(self):
        self.dxl = core.InterbotixRobotXSCore(self.config["robot_type"], self.config["name"], False)
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

    @register.inputs(tick=UInt64, action=Float32MultiArray)
    @register.outputs(action_applied=Float32MultiArray)
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Msg] = None,
    ):
        # Maps action=[0, 1.] to [-gripper_value, gripper_value].
        gripper_value = self.gripper.gripper_value * ((action.msgs[-1].data[0] * 2) - 1)
        # Set gripper value
        self.gripper.gripper_controller(gripper_value, delay=0.0)
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    def shutdown(self):
        self.gripper.open(delay=0.0)
