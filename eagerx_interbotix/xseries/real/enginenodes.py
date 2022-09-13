import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Any, Union
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register

from interbotix_copilot.client import Client
from interbotix_xs_modules import gripper, core


class XseriesSensor(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        arm_name: str,
        robot_type: str,
        joints: List[str],
        color: str = "cyan",
        mode: str = "position",
    ) -> NodeSpec:
        """Make the parameter specification for an Interbotix joint sensor.

        :param name: Node name
        :param rate: Rate of the node [Hz].
        :param arm_name: Name of the arm.
        :param robot_type: Manipulator type.
        :param joints: Joint names.
        :param color: Color of logged messages.
        :param mode: Types supported measurements:
                     - position: Joint positions [rad]. Should be limited by the joint limits.
                     - velocity: Angular velocity [rad/s]. Should be limited by the velocity limits.
                     - effort: Percentage of maximum allowed torque ~ [-1, 1].
                     - ee_position: End-effector position [m].
                     - ee_orientation: End-effector orientation [quaternion].
                     - ee_pose: End-effector pose [ee_position, ee_quaternion].
        :return:
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENGINE, color=color)
        spec.config.update(arm_name=arm_name, robot_type=robot_type)
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        spec.config.joints = joints

        # Update space definition based on mode.
        if mode == "effort":
            spec.outputs.obs.space.update(low=-1, high=1, shape=[len(joints)])
        elif mode == "ee_position":
            spec.outputs.obs.space.update(shape=[3])
        elif mode == "ee_orientation":
            spec.outputs.obs.space.update(low=-1, high=1, shape=[4])
        elif mode == "ee_pose":
            spec.outputs.obs.space.update(shape=[7])
        else:
            spec.outputs.obs.space.update(shape=[len(joints)])

        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.mode = spec.config.mode
        if self.mode not in ["position", "velocity", "effort", "ee_position", "ee_orientation", "ee_pose"]:
            raise NotImplementedError(f"This mode is not implemented: {spec.config.mode}")

        # Get arm client
        if "client" not in simulator:
            simulator["client"] = Client(spec.config.robot_type, spec.config.arm_name, group_name="arm")
        self.arm = simulator["client"]

        # Remap joint measurements & commands according to ordering in spec.config.joints.
        self.arm.set_joint_remapping(spec.config.joints)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(obs=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        # Select based on mode of node.
        if self.mode == "position":
            position = self.arm.get_joint_states().position
            obs = np.array(position, dtype="float32")
        elif self.mode == "velocity":
            velocity = self.arm.get_joint_states().velocity
            obs = np.array(velocity, dtype="float32")
        elif self.mode == "effort":
            effort = self.arm.get_joint_states().effort
            obs = np.array(effort, dtype="float32")
        elif self.mode == "ee_position":
            _, ee_position, _ = self.arm.get_ee_pose()
            obs = np.array(ee_position, dtype="float32")
        elif self.mode == "ee_orientation":
            rot_matrix, _, _ = self.arm.get_ee_pose()
            obs = R.from_matrix(rot_matrix).as_quat().astype("float32")
        elif self.mode == "ee_pose":
            rot_matrix, position, _ = self.arm.get_ee_pose()
            raise NotImplementedError("how to convert rotation_matrix into quaternion vector?")
            # obs = np.array(joint_state.effort, dtype="float32")
        else:
            raise NotImplementedError(f"This mode is not implemented: {self.mode}")
        return dict(obs=obs)

    def close(self):
        pass


class XseriesArm(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        arm_name: str,
        robot_type: str,
        joints: List[str],
        color: str = "green",
        mode: str = "position",
        profile_type: str = "time",
        profile_velocity: int = 2000,
        profile_acceleration: int = 300,
        kp_pos: Union[int, Dict[str, int]] = 800,
        ki_pos: Union[int, Dict[str, int]] = 0,
        kd_pos: Union[int, Dict[str, int]] = 0,
        kp_vel: Union[int, Dict[str, int]] = 100,
        ki_vel: Union[int, Dict[str, int]] = 1920,
        ff_acc: Union[int, Dict[str, int]] = 0,
        ff_vel: Union[int, Dict[str, int]] = 0,
    ) -> NodeSpec:
        """Make the parameter specification for an Interbotix controller.

        :param name: Name of the node
        :param rate: Rate of the node.
        :param arm_name: Name of the arm.
        :param robot_type: Manipulator type.
        :param joints: Names of all the joint motors in the group.
        :param color: Color of logged messages.
        :param mode: Either "position" or "velocity". Applies to the whole group of motors.
        :param profile_type: "time" or "velocity" (see InterbotixArm.set_operating_mode for info).
                             Applies to the whole group of motors.
        :param profile_velocity: Sets velocity of the Profile (see InterbotixArm.set_operating_mode for info).
                                 ‘0’ represents an infinite velocity. Applies to the whole group of motors.
        :param profile_acceleration: Sets acceleration time of the Profile (see InterbotixArm.set_operating_mode for info).
                                     ‘0’ represents an infinite acceleration. Applies to the whole group of motors.
        :param kp_pos: Position P gain. Either for a single motors (dict) or the whole group (int value).
        :param ki_pos: Position I gain. Either for a single motors (dict) or the whole group (int value).
        :param kd_pos: Position D gain. Either for a single motors (dict) or the whole group (int value).
        :param kp_vel: Velocity P gain. Either for a single motors (dict) or the whole group (int value).
        :param ki_vel: Velocity I gain. Either for a single motors (dict) or the whole group (int value).
        :param ff_acc: Feedforward 2nd gain. Either for a single motors (dict) or the whole group (int value).
        :param ff_vel: Feedforward 1st gain. Either for a single motors (dict) or the whole group (int value).
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENGINE, color=color)
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.joints = joints

        # Set motor configs
        spec.config.update(arm_name=arm_name, robot_type=robot_type)
        spec.config.update(
            mode=mode, profile_type=profile_type, profile_velocity=profile_velocity, profile_acceleration=profile_acceleration
        )
        spec.config.update(kp_pos=kp_pos, ki_pos=ki_pos, kd_pos=kd_pos)
        spec.config.update(kp_vel=kp_vel, ki_vel=ki_vel)
        spec.config.update(ff_acc=ff_acc, ff_vel=ff_vel)

        # Set shape of spaces
        spec.inputs.action.space.shape = [len(joints)]
        spec.outputs.action_applied.space.shape = [len(joints)]
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        # Get arm client
        if "client" not in simulator:
            simulator["client"] = Client(spec.config.robot_type, spec.config.arm_name, group_name="arm")
        self.arm = simulator["client"]

        # Remap joint measurements & commands according to ordering in spec.config.joints.
        self.arm.set_joint_remapping(spec.config.joints)

        # Set operating mode
        if spec.config.mode == "position":
            mode = Client.POSITION
        elif spec.config.mode == "velocity":
            mode = Client.VELOCITY
        else:
            raise ValueError(f"Mode `{spec.config.mode}` not recognized.")

        # Check control mode exists.
        if mode not in self.arm.SUPPORTED_MODES:
            raise NotImplementedError(f"The selected control mode is not implemented: {spec.config.mode}.")

        # Set operating mode
        self.arm.set_operating_mode(
            mode=mode,
            profile_type=spec.config.profile_type,
            profile_velocity=spec.config.profile_velocity,
            profile_acceleration=spec.config.profile_acceleration,
        )

        # Set gains
        gains = dict()
        gains["kp_pos"], gains["ki_pos"], gains["kd_pos"] = spec.config.kp_pos, spec.config.ki_pos, spec.config.kd_pos
        gains["kp_vel"], gains["ki_vel"] = spec.config.kp_vel, spec.config.ki_vel
        gains["ff_acc"], gains["ff_vel"] = spec.config.ff_acc, spec.config.ff_vel
        for key, gain in gains.items():
            if isinstance(gain, list):
                for name in spec.config.joints:
                    self.arm.set_pid_gains(name=name, **{key: gain})
            elif isinstance(gain, int):
                self.arm.set_pid_gains(**{key: gain})
            else:
                raise ValueError(f"Gain `{key}` has an invalid value `{gain}`.")

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg, action: Msg):
        # Get command
        cmd = action.msgs[-1]
        # Write command to arm
        self.arm.write_commands(cmd.tolist())
        # Send action that has been applied.
        return dict(action_applied=cmd)

    def shutdown(self):
        pass


class XseriesGripper(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        arm_name: str,
        robot_type: str,
        color: str = "green",
    ) -> NodeSpec:
        """XseriesGripper spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENGINE, color=color)
        spec.config.update(arm_name=arm_name, robot_type=robot_type)
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.dxl = core.InterbotixRobotXSCore(spec.config.robot_type, spec.config.name, False)
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
