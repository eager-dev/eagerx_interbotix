import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Any
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
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
        spec.config.inputs = ["orientation"]
        spec.config.outputs = ["yaw", "orientation"]
        spec.inputs.orientation.window = 0
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        pass

    @register.states()
    def reset(self):
        self.last_yaw = np.array(np.pi / 4, "float32")
        self.last_orientation = np.array([0, 0, 0, 1], dtype="float32")

    @register.inputs(orientation=Space(dtype="float32"))
    @register.outputs(
        yaw=Space(low=0, high=3.14 / 2, shape=(), dtype="float32"),
        orientation=Space(low=-1, high=1, shape=(4,), dtype="float32"),
    )
    def callback(self, t_n: float, orientation: Msg):
        if len(orientation.msgs) > 0:
            msgs = [orn for orn in orientation.msgs if len(orn) > 0]
            if len(msgs) == 0:  # Return if we received an empty message
                return dict(yaw=self.last_yaw, orientation=self.last_orientation)
            # In case of Pybullet, we must add a leading dummy dimension
            msgs = [orn[None, :] if orn.ndim == 1 else orn for orn in orientation.msgs]
            # Concatenate msgs
            orn = np.concatenate(msgs)
            rot = R.from_quat(orn).as_matrix()
            # Remove z axis from rotation matrix
            axis_idx = 2  # Upward pointing axis of robot base
            z_idx = np.argmax(np.abs(rot[:, axis_idx, :]), axis=1)  # Take absolute value, if axis points downward.
            rot_red = np.empty((rot.shape[0], 2, 2), dtype="float32")
            # Calculate angle
            for i, idx in enumerate(z_idx):
                s = np.sign(rot[i, axis_idx, idx])
                rot_red[i, :, :] = np.delete(np.delete(rot[i, :, :], obj=axis_idx, axis=0), obj=idx, axis=1)
                if (s > 0 and idx == 1) or (s < 0 and idx != 1):
                    rot_red[i, :, :] = rot_red[i, :, :] @ np.array([[0, 1], [1, 0]], dtype="float32")[None, :, :]
            th_cos = rot_red[:, 0, 0]
            th_sin = rot_red[:, 1, 0]
            th = np.arctan2(th_sin, th_cos)
            yaw = th % (np.pi / 2)
            self.last_yaw = np.mean(yaw, axis=0, dtype="float32")
            # self.backend.logwarn(f"[WrappedYawSensor] yaw={yaw}")
            # Calculate orientation
            rot_yaw = np.array(
                [
                    [np.cos(self.last_yaw), -np.sin(self.last_yaw), 0],
                    [np.sin(self.last_yaw), np.cos(self.last_yaw), 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            )
            self.last_orientation = R.from_matrix(rot_yaw).as_quat().astype("float32")

        return dict(yaw=self.last_yaw, orientation=self.last_orientation)
