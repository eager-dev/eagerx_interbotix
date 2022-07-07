from typing import List
from gym.spaces import Box, Discrete
import numpy as np
import eagerx
from eagerx.core.specs import ResetNodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg

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

        :param spec: Not provided by user.
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
        spec.inputs.joints.space = Box(low=np.array(lower, dtype="float32"), high=np.array(upper, dtype="float32"))
        spec.targets.goal.space = Box(low=np.array(lower, dtype="float32"), high=np.array(upper, dtype="float32"))
        spec.outputs.joints.space = Box(low=np.array(lower, dtype="float32"), high=np.array(upper, dtype="float32"))
        return spec

    def initialize(self, spec: ResetNodeSpec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout

        # Create a status publisher (not an output, because asynchronous)
        self.status_pub = self.backend.Publisher(f"{self.ns_name}/outputs/status", "uint64")

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(joints=None, in_collision=Discrete(3))
    @register.targets(goal=None)
    @register.outputs(joints=None, gripper=Box(low=np.array([0], dtype="float32"), high=np.array([1], dtype="float32")))
    def callback(self, t_n: float, goal: Msg = None, joints: Msg = None, in_collision: Msg = None):
        if self.start is None:
            self.start = t_n

        # todo: What to do if stuck in collision?
        if len(in_collision.msgs) > 0:
            in_collision = in_collision.msgs[-1].data
        else:
            in_collision = False

        # Process goal & current joint msgs
        joints = np.array(joints.msgs[-1].data, dtype="float32")
        goal = np.array(goal.msgs[-1].data, dtype="float32")

        # Determine done flag
        if np.isclose(joints, goal, atol=self.threshold).all():
            is_done = True
            self.status_pub.publish(np.array(status_map["Success"], dtype="int64"))
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
