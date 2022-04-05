from typing import Optional, List
import numpy as np

# IMPORT ROS
import rospy
from std_msgs.msg import Float32MultiArray, UInt64, Int64, Bool

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from eagerx.core.entities import ResetNode, SpaceConverter
from eagerx.core.constants import process as p

status_map = {"Success": 1, "Timeout": 2, "Collision": 3}


class ResetArm(ResetNode):
    @staticmethod
    @register.spec("ResetArm", ResetNode)
    def spec(
        spec,
        name: str,
        rate: float,
        upper: List[float],
        lower: List[float],
        gripper: bool = True,
        threshold: float = 0.02,
        timeout: float = 4.0,
        process: Optional[int] = p.NEW_PROCESS,
        color: Optional[str] = "grey",
    ):
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
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: BRIDGE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(ResetArm)

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

        # Add converter & space_converter
        spec.inputs.joints.space_converter = SpaceConverter.make("Space_Float32MultiArray", lower, upper, dtype="float32")
        spec.outputs.joints.space_converter = SpaceConverter.make("Space_Float32MultiArray", lower, upper, dtype="float32")
        spec.outputs.gripper.space_converter = SpaceConverter.make("Space_Float32MultiArray", [0], [1], dtype="float32")

    def initialize(self, threshold, timeout):
        self.threshold = threshold
        self.timeout = timeout

        # Create a status publisher (not an output, because asynchronous)
        self.status_pub = rospy.Publisher(f"{self.ns_name}/outputs/status", Int64, queue_size=0)

    @register.states()
    def reset(self):
        self.start = None

    @register.inputs(joints=Float32MultiArray, in_collision=UInt64)
    @register.targets(goal=Float32MultiArray)
    @register.outputs(joints=Float32MultiArray, gripper=Float32MultiArray)
    def callback(self, t_n: float, goal: Msg = None, joints: Msg = None, gripper: Msg = None, in_collision: Msg = None):
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
            self.status_pub.publish(Int64(data=status_map["Success"]))
        else:
            if self.timeout > 0 and self.timeout < (t_n - self.start):
                is_done = True
                if in_collision:
                    self.status_pub.publish(Int64(data=status_map["Collision"]))
                else:
                    self.status_pub.publish(Int64(data=status_map["Timeout"]))
            else:
                is_done = False

        # Create output message
        output_msgs = dict(joints=Float32MultiArray(data=goal), gripper=Float32MultiArray(data=[1.0]))
        output_msgs["goal/done"] = Bool(data=is_done)
        return output_msgs
