from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R
import eagerx
from eagerx import Space
from eagerx.core.specs import ResetNodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg
import modern_robotics as mr


status_map = {"Success": 1, "Timeout": 2, "Collision": 3}


class ResetArm(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        upper: List[float],
        lower: List[float],
        gripper: bool = True,
        threshold: float = 0.02,
        timeout: float = 4.0,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> ResetNodeSpec:
        """Resets joints & Gripper to goal positions.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param upper: Upper joint limits
        :param lower: Lower joint limits
        :param gripper: Include a gripper
        :param threshold: Closeness to the goal joints before considering the reset to be finished.
        :param timeout: Seconds before considering the reset to be finished, regardless of the closeness. A value of `0` means
                        indefinite.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["joints", "in_collision"]
        spec.config.targets = ["goal"]
        spec.config.outputs = ["joints"]
        # Add gripper of also controlled
        if gripper:
            spec.config.outputs.append("gripper")
        # Add custom params
        spec.config.threshold = threshold
        spec.config.timeout = timeout

        # Add variable space
        spec.inputs.joints.space.update(low=lower, high=upper)
        spec.targets.goal.space.update(low=lower, high=upper)
        spec.outputs.joints.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout

        # Create a status publisher (not an output, because asynchronous)
        self.status_pub = self.backend.Publisher(f"{self.ns_name}/outputs/status", "uint64")

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(joints=Space(dtype="float32"), in_collision=Space(low=0, high=2, shape=(), dtype="int64"))
    @register.targets(goal=Space(dtype="float32"))
    @register.outputs(joints=Space(dtype="float32"), gripper=Space(low=[0.0], high=[1.0]))
    def callback(self, t_n: float, goal: Msg = None, joints: Msg = None, in_collision: Msg = None):
        if self.start is None:
            self.start = t_n

        # todo: What to do if stuck in collision?
        if len(in_collision.msgs) > 0:
            in_collision = in_collision.msgs[-1].data
        else:
            in_collision = False

        # Process goal & current joint msgs
        joints = joints.msgs[-1]
        goal = goal.msgs[-1]

        # Determine done flag
        if np.isclose(joints, goal, atol=self.threshold).all():
            is_done = True
            self.status_pub.publish(np.array(status_map["Success"], dtype="uint64"))
        else:
            if self.timeout > 0 and self.timeout < (t_n - self.start):
                is_done = True
                if in_collision:
                    self.status_pub.publish(np.array(status_map["Collision"], dtype="uint64"))
                else:
                    self.status_pub.publish(np.array(status_map["Timeout"], dtype="uint64"))
            else:
                is_done = False

        # Create output message
        output_msgs = dict(joints=goal, gripper=np.array([1.0], dtype="float32"))
        output_msgs["goal/done"] = is_done
        return output_msgs


class MoveUp(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        Slist: List[List[float]],
        M: List[List[float]],
        target_pos: List[float],
        timeout: float = 5.0,
        pos_threshold: float = 0.05,
        ori_threshold: float = 0.25,
        pos_gain: float = 1.0,
        ori_gain: float = 1.0,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> ResetNodeSpec:
        """Resets joints to goal positions.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param Slist: Screw axes in the space frame when the manipulator is at the home position, in the format of a matrix
                        with the screw axes as the columns.
        :param M: The home configuration of the end-effector.
        :param target_pos: Target joint positions.
        :param timeout: Seconds before considering the reset to be finished, regardless of the closeness. A value of `0` means
                        indefinite.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["ee_position", "ee_orientation"]
        spec.config.targets = ["velocity"]
        spec.config.outputs = ["dxyz", "dyaw"]

        # Add custom params
        spec.config.Slist = Slist
        spec.config.M = M
        spec.config.target_pos = target_pos
        spec.config.pos_threshold = pos_threshold
        spec.config.ori_threshold = ori_threshold
        spec.config.pos_gain = pos_gain
        spec.config.ori_gain = ori_gain
        spec.config.timeout = timeout
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.timeout = spec.config.timeout
        self.pos_threshold = spec.config.pos_threshold
        self.ori_threshold = spec.config.ori_threshold
        self.pos_gain = spec.config.pos_gain
        self.ori_gain = spec.config.ori_gain
        self.Slist = np.array(spec.config.Slist)
        self.M = np.array(spec.config.M)
        self.target_pos = np.array(spec.config.target_pos)

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(ee_position=Space(dtype="float32"), ee_orientation=Space(dtype="float32"))
    @register.targets(velocity=Space(dtype="float32"))
    @register.outputs(dxyz=Space(dtype="float32"), dyaw=Space(dtype="float32"))
    def callback(self, t_n: float, velocity: Msg = None, ee_position: Msg = None, ee_orientation: Msg = None):
        if self.start is None:
            self.start = t_n

        # Process goal & current joint msgs
        ee_position = ee_position.msgs[-1]
        orn = ee_orientation.msgs[-1]
        rot_ee2b = R.from_quat(orn).as_matrix()
        rot_red_ee2b = rot_ee2b[:2, 1:]
        yaw = np.arctan2(rot_red_ee2b[1, 1], rot_red_ee2b[0, 1])

        transformation = mr.FKinSpace(self.M, self.Slist, self.target_pos)
        goal_xyz = transformation[:3, 3]

        # Determine done flag
        dxyz = (goal_xyz - ee_position) * self.pos_gain
        dyaw = -yaw * self.ori_gain
        is_done = False
        if np.isclose(ee_position, goal_xyz, atol=self.pos_threshold).all() and np.isclose(yaw, 0, atol=self.ori_threshold):
            dxyz *= 0
            dyaw *= 0
            is_done = True
        if self.timeout > 0 and self.timeout < (t_n - self.start):
            dxyz *= 0
            dyaw *= 0
            is_done = True

        # Create output message
        output_msgs = dict(dxyz=np.asarray(dxyz, dtype="float32"), dyaw=np.asarray(dyaw, dtype="float32"))
        output_msgs["velocity/done"] = is_done
        return output_msgs


class MoveUpVelControl(eagerx.ResetNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        target_pos: List[float],
        timeout: float = 5.0,
        threshold: float = 0.05,
        pos_gain: float = 0.5,
        vel_gain: float = 0.1,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> ResetNodeSpec:
        """Resets joints to goal positions.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param Slist: Screw axes in the space frame when the manipulator is at the home position, in the format of a matrix
                        with the screw axes as the columns.
        :param M: The home configuration of the end-effector.
        :param target_pos: Target joint positions.
        :param timeout: Seconds before considering the reset to be finished, regardless of the closeness. A value of `0` means
                        indefinite.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["current_pos", "current_vel"]
        spec.config.targets = ["velocity"]
        spec.config.outputs = ["joint_vel"]

        # Add custom params
        spec.config.target_pos = target_pos
        spec.config.threshold = threshold
        spec.config.pos_gain = pos_gain
        spec.config.vel_gain = vel_gain
        spec.config.timeout = timeout
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.timeout = spec.config.timeout
        self.threshold = spec.config.threshold
        self.pos_gain = spec.config.pos_gain
        self.vel_gain = spec.config.vel_gain
        self.target_pos = np.array(spec.config.target_pos)

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(current_pos=Space(dtype="float32"), current_vel=Space(dtype="float32"))
    @register.targets(velocity=Space(dtype="float32"))
    @register.outputs(joint_vel=Space(dtype="float32"))
    def callback(self, t_n: float, velocity: Msg = None, current_pos: Msg = None, current_vel: Msg = None):
        if self.start is None:
            self.start = t_n

        # Process goal & current joint msgs
        joint_pos = current_pos.msgs[-1]
        joint_vel = current_vel.msgs[-1]

        # Determine done flag

        vel_command = (self.target_pos - joint_pos) * self.pos_gain - joint_vel * self.vel_gain

        is_done = False
        if np.isclose(joint_pos, self.target_pos, atol=self.threshold).all():
            vel_command *= 0
            is_done = True
        if self.timeout > 0 and self.timeout < (t_n - self.start):
            vel_command *= 0
            is_done = True

        # Create output message
        output_msgs = dict(joint_vel=np.asarray(vel_command, dtype="float32"))
        output_msgs["velocity/done"] = is_done
        return output_msgs
