import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
import eagerx.core.register as register
from eagerx_interbotix.aruco_detector import ArucoPoseDetector


class CameraCalibrator(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        aruco_id: int,
        aruco_size: float,
        aruco_type: str,
        marker_translation: List[float],
        marker_rotation: List[float],
        cam_intrinsics: Dict,
        process: int = eagerx.NEW_PROCESS,
    ) -> NodeSpec:
        """Make the parameter specification for an Goal observation sensor.

        :param name: Node name.
        :param rate: Rate of the node [Hz].
        :param aruco_id: Unique identifier of the Aruco marker.
        :param aruco_size: Aruco marker size [m].
        :param aruco_type: type of aruco dict (e.g. `DICT_ARUCO_ORIGINAL`, `DICT_5X5_100`).
        :param marker_translation: Translation marker to ee frame [m].
        :param marker_rotation: Rotation marker to ee frame [quaternion].
        :param cam_intrinsics: A dict with the intrinsic parameters of the camera.
                               Has the format: wiki.ros.org/camera_calibration.
        :param process: Process to launch node in.
        :return: Parameter specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process)
        spec.config.inputs = ["image", "ee_position", "ee_orientation"]
        spec.config.outputs = ["translation", "orientation", "image_aruco"]
        spec.config.aruco_id = aruco_id
        spec.config.aruco_size = aruco_size
        spec.config.aruco_type = aruco_type
        spec.config.marker_translation = marker_translation
        spec.config.marker_rotation = marker_rotation
        spec.config.cam_intrinsics = cam_intrinsics
        return spec

    def initialize(self, spec: NodeSpec):
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
        marker_trans = spec.config.marker_translation
        marker_quat = spec.config.marker_rotation
        self.T_a2ee = np.zeros((4, 4), dtype="float32")
        self.T_a2ee[3, 3] = 1
        self.T_a2ee[:3, :3] = R.from_quat(marker_quat).as_matrix()
        self.T_a2ee[:3, 3] = marker_trans

    def reset(self):
        pass

    @register.inputs(
        image=Space(dtype="uint8"),
        ee_position=Space(shape=(3,), dtype="float32"),
        ee_orientation=Space(shape=(4,), dtype="float32"),
    )
    @register.outputs(
        translation=Space(low=-1, high=1, shape=(3,), dtype="float32"),
        orientation=Space(low=-1, high=1, shape=(4,), dtype="float32"),
        image_aruco=Space(dtype="uint8"),
    )
    def callback(self, t_n: float, image: Msg, ee_position: Msg, ee_orientation: Msg):
        image_raw = image.msgs[-1]

        # Undistort image
        image = self.detector.undistort(image_raw)

        # Get pose
        image, corners, ids, rvec, tvec = self.detector.estimate_pose(image, draw=True)

        # Get pose measurements (filter for aurco_id)
        if (
            rvec is not None
            and (ids == self.aruco_id)[:, 0].any()
            and len(ee_position.msgs) > 0
            and len(ee_orientation.msgs) > 0
        ):
            mask = (ids == self.aruco_id)[:, 0]
            rvec = rvec[mask]
            tvec = tvec[mask]

            # Get position & orientation
            ee_position = ee_position.msgs[-1]
            ee_orientation = ee_orientation.msgs[-1]

            # Create cam2aruco transformation (inv(T_a2c).
            T_c2a = np.zeros((4, 4), dtype="float32")
            T_c2a[3, 3] = 1
            T_c2a[:3, :3] = R.from_rotvec(rvec[0, 0, :]).as_matrix().transpose()
            T_c2a[:3, 3] = -T_c2a[:3, :3] @ tvec[0, 0, :]
            # T_c2a = np.linalg.inv(T_a2c)  # todo: verify this is correct.
            # Create ee2base transformation (T_ee2b)
            T_ee2b = np.zeros((4, 4), dtype="float32")
            T_ee2b[3, 3] = 1
            T_ee2b[:3, :3] = R.from_quat(ee_orientation).as_matrix()
            T_ee2b[:3, 3] = ee_position
            # Calculate cam2base transformation (T_c2b)
            T_c2b = T_ee2b @ self.T_a2ee @ T_c2a
            translation_c2b = T_c2b[:3, 3].astype("float32")
            orientation_c2b = R.from_matrix(T_c2b[:3, :3]).as_quat().astype("float32")
        else:
            # Output empty orientation if nothing was detected.
            orientation_c2b = np.zeros((4,), dtype="float32")
            translation_c2b = np.zeros((3,), dtype="float32")
        return dict(translation=translation_c2b, orientation=orientation_c2b, image_aruco=image)

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
