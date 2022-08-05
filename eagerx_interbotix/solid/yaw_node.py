import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec, ObjectSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register


class WrappedYawSensor(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.NEW_PROCESS)
        spec.config.inputs = ["tick", "orientation"]
        spec.config.outputs = ["obs"]

        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        pass

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"), orientation=Space(low=-1, high=1, shape=(4,), dtype="float32"))
    @register.outputs(obs=Space(low=0, high=3.14 / 2, shape=(), dtype="float32"))
    def callback(self, t_n: float, tick: Msg, orientation: Msg):
        rot = R.from_quat(orientation.msgs[-1]).as_matrix()
        # remove z axis from rotation matrix
        z_idx = np.argmax(np.abs(rot[2, :]))  # Take absolute value, if axis points downward).
        rot_red = np.delete(np.delete(rot, obj=z_idx, axis=0), obj=z_idx, axis=1)
        # calculate angle
        acos = np.arccos(rot_red)
        acos_wrapped = acos - (np.pi / 2) * np.floor(acos / (np.pi / 2))
        yaw = acos_wrapped[0, 0]  # todo: why not all the same value?
        return dict(obs=yaw)
