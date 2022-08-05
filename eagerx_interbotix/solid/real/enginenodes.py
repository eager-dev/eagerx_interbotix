import numpy as np
from typing import Dict, List, Any, Union
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec, ObjectSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register


class GoalObservationSensor(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        color: str = "cyan",
        mode: str = "position",
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :param color: Color of logged messages.
        :param mode: Type of observation (e.g. `position`, `orientation`).
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENGINE, color=color)
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = mode
        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        self.mode = spec.config.mode
        self.simulator = simulator
        self.obj_name = object_spec.config.name

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(obs=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        # Get measurement of joint state
        obs = self.simulator[self.obj_name][self.mode]
        return dict(obs=obs)

    def close(self):
        pass


class ArucoPoseDetector(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        aruco_id: int,
        aruco_size: int,
        aruco_type: str,
        aruco_offset: List[float],
        cam_translation: List[float],
        cam_rotation: List[float],
        color: str = "cyan",
        process: int = eagerx.NEW_PROCESS,
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :param aruco_id: Unique identifier of the Aruco marker.
        :param aruco_size: Aruco marker size [mm].
        :param aruco_type: type of aruco dict (e.g. `DICT_ARUCO_ORIGINAL`, `DICT_5X5_100`).
        :param aruco_offset: Offset from Aruco marker pose to object cog in Aruco frame [m].
        :param cam_translation: Translation to camera in base frame (origin base to camera) [m].
        :param cam_rotation: Rotation to camera in base frame (origin base to camera) [quaternion].
        :param color: Color of logged messages.
        :param process: Process to launch node in.
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["image"]
        spec.config.outputs = ["position", "orientation", "image_aruco"]
        spec.config.aruco_id = aruco_id
        spec.config.aruco_size = aruco_size
        spec.config.aruco_type = aruco_type
        spec.config.aruco_offset = aruco_offset
        return spec

    def initialize(self, spec: NodeSpec, object_spec: ObjectSpec, simulator: Any):
        # todo: initialize detector
        self.aruco_id = spec.config.aruco_id
        self.aruco_size = spec.config.aruco_size
        self.aruco_type = spec.config.aruco_type
        self.aruco_offset = spec.config.aruco_size
        # todo: calculate cam_to_base transformation matrix here
        self.cam_translation = spec.config.aruco_size
        self.cam_rotation = spec.config.aruco_size

    @register.states()
    def reset(self):
        pass

    @register.inputs(image=Space(dtype="uint8"))
    @register.outputs(position=Space(shape=(3,), dtype="float32"),
                      orientation=Space(shape=(4,), dtype="float32"),
                      image_aruco=Space(dtype="uint8"))
    def callback(self, t_n: float, image: Msg):
        # todo: Get aruco pose by passing image through aruco pose detector
        # todo: Transform pose with cam_to_base transformation.
        # todo: Plot aruco pose on image
        # todo: what if nothing detected? Output last pose.
        img = image.msgs[-1]
        position = np.array([0.4, -0.2, 0.05], dtype="float32")
        orientation = np.array([0, 0, 0, 1], dtype="float32")
        return dict(position=position, orientation=orientation, image_aruco=image.msgs[-1])

    def close(self):
        pass