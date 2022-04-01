from typing import Optional, List
import numpy as np

# IMPORT ROS
from std_msgs.msg import Float32MultiArray

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from eagerx.core.entities import Node, SpaceConverter
from eagerx.core.constants import process as p


class SafetyFilter(Node):
    @staticmethod
    @register.spec("SafetyFilter", Node)
    def spec(
        spec,
        name: str,
        rate: float,
        joints: List[int],
        upper: List[float],
        lower: List[float],
        vel_limit: List[float],
        duration: Optional[float] = None,
        checks: int = 3,
        process: Optional[int] = p.NEW_PROCESS,
        color: Optional[str] = "grey",
    ):
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param spec: Not provided by user.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :param joints: joint names
        :param upper: upper joint limits
        :param lower: lower joint limits
        :param vel_limit: absolute velocity joint limits
        :param duration: time (seconds) it takes to reach the commanded positions from the current position.
        :param checks: collision checks performed over the duration.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: BRIDGE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(SafetyFilter)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["goal", "current"]
        spec.config.outputs = ["filtered"]

        # Modify custom node params
        spec.config.joints = joints
        spec.config.upper = upper
        spec.config.lower = lower
        spec.config.vel_limit = vel_limit
        spec.config.duration = duration if isinstance(duration, float) else 2. / rate
        spec.config.checks = checks

        # Add converter & space_converter
        spec.inputs.goal.space_converter = SpaceConverter.make("Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32")
        spec.inputs.current.space_converter = SpaceConverter.make("Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32")
        spec.outputs.filtered.space_converter = SpaceConverter.make("Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32")

    def initialize(self, joints, upper, lower, vel_limit, duration, checks):
        self.joints = joints
        self.upper = np.array(upper, dtype="float")
        self.lower = np.array(lower, dtype="float")
        self.vel_limit = np.array(vel_limit, dtype="float")
        self.duration = duration
        self.checks = checks
        self.dt = 1 /self.rate

    @register.states()
    def reset(self):
        pass

    @register.inputs(goal=Float32MultiArray, current=Float32MultiArray)
    @register.outputs(filtered=Float32MultiArray)
    def callback(self, t_n: float, goal: Msg = None, current: Msg = None):
        goal = np.array(goal.msgs[-1].data, dtype="float32")
        current = np.array(current.msgs[-1].data, dtype="float32")

        # Clip to joint limits
        filtered = np.clip(goal, self.lower, self.upper)

        # Linearly interpolate
        t = np.linspace(self.dt, self.dt+self.duration, self.checks)
        interp = np.empty((current.shape[0], self.checks), dtype="float32")
        for i in range(current.shape[0]):
            interp[i][:] = np.interp(t, [0, self.dt+self.duration], [current[i], filtered[i]])

        # Reduce goal to vel_limit
        diff = filtered - current
        too_fast = np.abs(diff / (self.dt+self.duration)) > self.vel_limit
        if np.any(too_fast):
            filtered[too_fast] = current[too_fast] + np.sign(diff[too_fast]) * (self.dt+self.duration)* self.vel_limit[too_fast]

        # todo: collision check with environment
        # todo: self_colision check

        return dict(filtered=Float32MultiArray(data=filtered))
