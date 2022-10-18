import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Any
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register
from eagerx_interbotix.aruco_detector import ArucoPoseDetector


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

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.mode = spec.config.mode
        self.simulator = simulator

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(obs=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        # Get measurement of joint state
        obs = self.simulator[self.mode]
        return dict(obs=obs)

    def close(self):
        pass


class YawGoalObservationSensor(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        color: str = "cyan",
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :param color: Color of logged messages.
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=eagerx.ENGINE, color=color)
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.mode = "orientation"
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.mode = spec.config.mode
        self.simulator = simulator

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"))
    @register.outputs(obs=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Msg):
        # Get measurement of joint state
        orn = self.simulator[self.mode]
        rot = R.from_quat(orn)
        yaw = rot.as_euler("zyx", degrees=True)[0]
        yaw = yaw % (np.pi / 2)
        return dict(obs=np.asarray([yaw], dtype="float32"))

    def close(self):
        pass


class PoseDetector(eagerx.EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        aruco_id: int,
        aruco_size: float,
        aruco_type: str,
        object_translation: List[float],
        cam_translation: List[float],
        cam_rotation: List[float],
        cam_intrinsics: Dict,
        color: str = "cyan",
        process: int = eagerx.NEW_PROCESS,
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :param aruco_id: Unique identifier of the Aruco marker.
        :param aruco_size: Aruco marker size [m].
        :param aruco_type: type of aruco dict (e.g. `DICT_ARUCO_ORIGINAL`, `DICT_5X5_100`).
        :param object_translation: Translation from Aruco marker pose to object cog in Aruco frame [m].
        :param cam_translation: Translation to camera in base frame (origin base to camera) [m].
        :param cam_rotation: Rotation to camera in base frame (origin base to camera) [quaternion].
        :param cam_intrinsics: A dict with the intrinsic parameters of the camera.
                               Has the format: wiki.ros.org/camera_calibration.
        :param color: Color of logged messages.
        :param process: Process to launch node in.
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["image"]
        spec.config.outputs = ["position", "orientation", "image_aruco"]
        spec.config.states = ["position"]
        spec.config.aruco_id = aruco_id
        spec.config.aruco_size = aruco_size
        spec.config.aruco_type = aruco_type
        spec.config.object_translation = object_translation
        spec.config.cam_translation = cam_translation
        spec.config.cam_rotation = cam_rotation
        spec.config.cam_intrinsics = cam_intrinsics
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        self.aruco_id = spec.config.aruco_id
        ci = spec.config.cam_intrinsics

        # Initialize detector
        aruco_size = spec.config.aruco_size
        aruco_type = spec.config.aruco_type
        height, width = ci["image_height"], ci["image_width"]
        camera_matrix = np.array(ci["camera_matrix"]["data"], dtype="float32").reshape(
            ci["camera_matrix"]["rows"], ci["camera_matrix"]["cols"]
        )
        dist_coeffs = np.array(ci["distortion_coefficients"]["data"], dtype="float32").reshape(
            ci["distortion_coefficients"]["rows"], ci["distortion_coefficients"]["cols"]
        )
        self.detector = ArucoPoseDetector(height, width, aruco_size, camera_matrix, dist_coeffs, aruco_type)

        # Calculate cam_to_base transformation matrix
        cam_trans = spec.config.cam_translation
        cam_quat = spec.config.cam_rotation
        self.T_c2b = np.zeros((4, 4), dtype="float32")
        self.T_c2b[3, 3] = 1
        self.T_c2b[:3, :3] = R.from_quat(cam_quat).as_matrix()
        self.T_c2b[:3, 3] = cam_trans

        # Object translation offset
        t = spec.config.object_translation
        self.object_translation = np.array([[t[0]], [t[1]], [t[2]], [1.0]], dtype="float32")

    @register.states(position=Space(shape=(3,), dtype="float32"))
    def reset(self, position: np.ndarray):
        self.pos_last = position

    @register.inputs(image=Space(dtype="uint8"))
    @register.outputs(position=Space(dtype="float32"), orientation=Space(dtype="float32"), image_aruco=Space(dtype="uint8"))
    def callback(self, t_n: float, image: Msg):
        image_raw = image.msgs[-1]

        # Undistort image
        image = self.detector.undistort(image_raw)

        # Get pose
        image, corners, ids, rvec, tvec = self.detector.estimate_pose(image, draw=True)

        # Get pose measurements (filter for aurco_id)
        if rvec is not None and (ids == self.aruco_id)[:, 0].any():
            mask = (ids == self.aruco_id)[:, 0]
            rvec = rvec[mask]
            tvec = tvec[mask]
            # Position aruco (with offset) in camera frame (aic).
            pos_aic = self._apply_object_translation(rvec, tvec)
            # Position aruco (with offset) in base frame (aib).
            pos_aib = self.T_c2b[:3, :3] @ pos_aic + self.T_c2b[:3, 3, None]
            # Rotation matrix from aruco to base frame
            rmat_a2c = R.from_rotvec(rvec[:, 0, :]).as_matrix()
            rmat_a2b = self.T_c2b[:3, :3] @ rmat_a2c
            quat_a2b = R.from_matrix(rmat_a2b).as_quat().astype("float32")
            # Store last position
            orientation = quat_a2b
            self.pos_last = np.mean(pos_aib, axis=0)[:, 0]
            # self.backend.logwarn(f"[PoseDetector] x={pos_aib[:, 0, 0]} | y={pos_aib[:, 1, 0]}, z={pos_aib[:, 2, 0]}")
        else:
            # Output empty orientation if nothing was detected.
            orientation = np.zeros((0, 4), dtype="float32")
        position = self.pos_last
        return dict(position=position, orientation=orientation, image_aruco=image)

    def _apply_object_translation(self, rvec, tvec):
        T_a2c = np.zeros((rvec.shape[0], 4, 4), dtype="float32")
        T_a2c[:, 3, 3] = 1
        # Set all rotation matrices
        rmat = R.from_rotvec(rvec[:, 0, :])
        T_a2c[:, :3, :3] = rmat.as_matrix()
        # Set all translations
        T_a2c[:, :3, 3] = tvec[:, 0, :]
        # Offset measurements
        position = (T_a2c @ self.object_translation[:, :])[:, :3, :]
        return position

    def close(self):
        pass
