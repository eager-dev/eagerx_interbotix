import typing as t
from scipy.spatial.transform import Rotation as R
import numpy as np
import modern_robotics as mr
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg
from math import ceil


class SmithPredictor(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        Slist: t.List[t.List[float]],
        M: t.List[t.List[float]],
        delays: t.Union[t.List[float], float],
        motor_constants: t.Union[t.List[float], float],
        upper: t.List[float],
        lower: t.List[float],
        vel_limit: t.List[float],
        process: int = eagerx.NEW_PROCESS,
        log_level: int = eagerx.INFO,
    ) -> NodeSpec:
        """
        Smith predictor for delay compensation.

        :param name: Node name.
        :param rate: Rate at which callback is called.
        :param Slist: The joint screw axes in the space frame when the manipulator is at the home position,
                      in the format of a matrix with axes as the columns. See modern robotics toolkit.
        :param M: The home configuration of the end-effector.
        :param delays: Delays of the first order model. Can be a list of delays for each joint.
        :param motor_constants: Motor constants of the first order model. Can be a list of motor constants for each joint.
        :param upper: Upper joint limits.
        :param lower: Lower joint limits.
        :param vel_limit: Velocity limits.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}.
        :return: Parameter specification of the node.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["vel_command", "pos", "vel"]
        spec.config.outputs = ["ee_orn", "ee_pos", "pos", "vel"]
        spec.config.log_level = log_level

        # Modify custom node params
        spec.config.Slist = Slist
        spec.config.M = M
        spec.config.delays = delays if isinstance(delays, list) else [delays] * len(Slist)
        spec.config.motor_constants = motor_constants if isinstance(motor_constants, list) else [motor_constants] * len(Slist)

        # Modify input spaces
        vel_limit = vel_limit if isinstance(vel_limit, list) else [vel_limit] * len(Slist)
        vel_limit = np.array(vel_limit, dtype="float32")
        spec.inputs.vel_command.space = Space(dtype="float32", shape=(len(Slist),), low=-vel_limit, high=vel_limit)
        spec.inputs.pos.space = Space(dtype="float32", shape=(len(Slist),), low=lower, high=upper)
        spec.inputs.vel.space = Space(dtype="float32", shape=(len(Slist),), low=-vel_limit, high=vel_limit)

        # Modify output spaces
        spec.outputs.ee_orn.space = Space(dtype="float32", shape=(4,), low=-1, high=1)
        spec.outputs.ee_pos.space = Space(dtype="float32", shape=(3,))
        spec.outputs.pos.space = Space(dtype="float32", shape=(len(Slist),), low=lower, high=upper)
        spec.outputs.vel.space = Space(dtype="float32", shape=(len(Slist),), low=-vel_limit, high=vel_limit)
        return spec

    def initialize(self, spec: NodeSpec):
        np.set_printoptions(precision=2, suppress=True)
        self.Slist = np.array(spec.config.Slist, dtype="float32")
        self.M = np.array(spec.config.M, dtype="float32")
        self.delays = np.array(spec.config.delays, dtype="float32")
        self.motor_constants = np.array(spec.config.motor_constants, dtype="float32")
        self.pos_predictions = []

    @register.states()
    def reset(self):
        pass

    @register.inputs(
        vel_command=Space(dtype="float32"),
        pos=Space(dtype="float32"),
        vel=Space(dtype="float32"),
    )
    @register.outputs(
        ee_orn=Space(dtype="float32"),
        ee_pos=Space(dtype="float32"),
        pos=Space(dtype="float32"),
        vel=Space(dtype="float32"),
    )
    def callback(self, t_n: float, vel_command: Msg, pos: Msg, vel: Msg):
        pos_msg = pos.msgs[-1]
        vel_msg = vel.msgs[-1]
        if len(vel_command.msgs) > 0:
            vel_command_msg = vel_command.msgs[-1]
            # Predict joint velocities
            vel_predicted = vel_command_msg + (vel_msg - vel_command_msg) * np.exp(-self.delays / self.motor_constants)

            # Predict joint positions
            pos_predicted = (
                pos_msg
                + vel_command_msg * self.delays
                + (vel_msg - vel_command_msg)
                * (self.delays - self.delays * np.exp(-self.delays / self.motor_constants))
                / self.motor_constants
            )
            pos_diff_predicted = np.abs(pos_predicted - pos_msg)
            self.pos_predictions.append(pos_diff_predicted)
        else:
            vel_predicted = vel_msg
            pos_predicted = pos_msg
        # if len(pos.msgs) > 1 + round(max(self.delays) * self.rate):
        #     pos_diff = abs(pos.msgs[-1] - pos.msgs[-1 - round(max(self.delays) * self.rate)])
        #     pos_diff_predicted = self.pos_predictions[-1 - round(max(self.delays) * self.rate)]
        #     error = (pos_diff - pos_diff_predicted)
        #     self.motor_constants -= 1 * error
        #     self.backend.logwarn(f"Constants: {self.motor_constants}")
        transformation = mr.FKinSpace(self.M, self.Slist, pos_predicted)
        ee_orn_predicted = R.from_matrix(transformation[:3, :3]).as_quat()
        ee_pos_predicted = transformation[:3, 3]
        return dict(ee_orn=ee_orn_predicted, ee_pos=ee_pos_predicted, pos=pos_predicted, vel=vel_predicted)
