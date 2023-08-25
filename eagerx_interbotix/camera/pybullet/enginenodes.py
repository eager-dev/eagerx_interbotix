from typing import Optional, List, Dict
from scipy.spatial.transform import Rotation as R
import cv2
import pybullet

# IMPORT EAGERX
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.core.constants import process as p
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class CameraSensor(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: Optional[int] = p.ENGINE,
        color: Optional[str] = "cyan",
        mode: str = "rgb",
        inputs: List[str] = None,
        render_shape: List[int] = None,
        fov: float = 57.0,
        near_val: float = 0.1,
        far_val: float = 10.0,
        debug: bool = False,
        states: List[str] = None,
    ):
        """A spec to create a CameraSensor node that provides images that can be used for perception and/or rendering.

        For more info on `fov`, `near_val`, and `far_val`, see the `Synthetic Camera Rendering` section in
        https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.33wr3gwy5kuj

        It is considered that:
         - the position is the camera eye position in Cartesian world coordinates.
         - the positive z-axis of the camera pose in Cartesian world coordinates points to the camera target.
         - The negative y-axis of the camera pose in Cartesian world coordinates points upward in the image.

        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :param color: Specifies the color of logged messages & node color in the GUI.
        :param mode: Available: `rgb`, `bgr`, `rgbd`, `bgrd`, `bgra` and `rgba`.
        :param inputs: Optionally, if the camera pose changes over time select `pos` and/or `orientation` & connect
                       accordingly, to dynamically change the camera view. If not selected, `pos` and/or `orientation` are
                       selected as states. This means you can choose a static camera pose at the start of every episode.
        :param render_shape: The shape of the produced images [height, width].
        :param fov: Field of view.
        :param near_val: Near plane distance [m].
        :param far_val: Far plane distance [m].
        :param debug: True will plot the camera pose using debug lines.
        :return: NodeSpec
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = inputs if isinstance(inputs, list) else ["tick"]
        spec.config.outputs = ["image"]

        # Add states if position and orientation are not inputs.
        spec.config.states = states if states else []
        if not states:
            if "pos" not in spec.config.inputs:
                spec.config.states.append("pos")
            if "orientation" not in spec.config.inputs:
                spec.config.states.append("orientation")

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.update(mode=mode, fov=fov, near_val=near_val, far_val=far_val)
        spec.config.flags = pybullet.ER_NO_SEGMENTATION_MASK
        spec.config.renderer = pybullet.ER_BULLET_HARDWARE_OPENGL
        spec.config.render_shape = render_shape if isinstance(render_shape, list) else [480, 680]
        spec.config.debug = debug

        # Position
        spec.states.pos.space = Space(low=[-5, -5, 0], high=[5, 5, 5])

        # Orientation
        spec.states.orientation.space = Space(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])

        # Light Direction
        spec.states.light_direction.space = Space(low=[0, 0, 0], high=[1, 1, 1])

        # Image
        channels = 3 if mode in ["rgb", "bgr"] else 4
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], channels)
        spec.outputs.image.space = Space(low=0, high=255, shape=shape, dtype="uint8")
        return spec

    def initialize(self, spec: NodeSpec, simulator: Dict):
        """Initializes the camera sensor according to the spec."""
        if simulator:
            self._p = simulator["client"]
        else:
            raise NotImplementedError("Currently, rendering leads to an error when connecting via shared memory.")
            # from pybullet_utils import bullet_client
            # self._p = bullet_client.BulletClient(pybullet.SHARED_MEMORY, options="-shared_memory_key 1234")
            # self._p = pybullet.connect(pybullet.SHARED_MEMORY, key=1234)
        # print("[rgb]: ", self._p._client)
        self.mode = spec.config.mode
        self.debug = spec.config.debug
        self.x_axis_id = None
        self.y_axis_id = None
        self.z_axis_id = None
        self.height, self.width = spec.config.render_shape
        self.intrinsic = dict(
            fov=spec.config.fov, nearVal=spec.config.near_val, farVal=spec.config.far_val, aspect=self.height / self.width
        )
        self.cb_args = dict(
            width=self.width,
            height=self.height,
            viewMatrix=None,
            projectionMatrix=None,
            flags=spec.config.flags,
            renderer=spec.config.renderer,
            physicsClientId=self._p._client,
            lightDirection=None,
        )
        self.cam_cb = self._camera_measurement(self._p, self.mode, self.cb_args)

    @register.states(pos=Space(dtype="float32"), orientation=Space(dtype="float32"), light_direction=Space(dtype="float32"))
    def reset(self, pos=None, orientation=None, light_direction=None):
        """The static position and orientation of the camera sensor can be reset at the start of a new episode.

        If 'position' and 'orientation' were selected as inputs in the spec, nothing happens here because the camera pose
        changes over time according to the connected inputs.
        """
        self.cb_args["projectionMatrix"] = pybullet.computeProjectionMatrixFOV(**self.intrinsic)

        if pos is not None and orientation is not None:
            if self.debug:
                self._debug_plot_camera_pose(pos, orientation, self._p)
            self.cb_args["viewMatrix"] = self._view_matrix(pos, orientation, self._p._client)
        if pos is not None:
            self.pos = pos
        if orientation is not None:
            self.orientation = orientation
        if light_direction is not None:
            self.cb_args["lightDirection"] = light_direction.tolist()

    def _debug_plot_camera_pose(self, position, orientation, p):
        rot = R.from_quat(orientation).as_matrix()
        lineFromXYZ = position
        x_axis = (0.1 * rot[:, 0] + position, [1, 0, 0])
        y_axis = (0.1 * rot[:, 1] + position, [0, 1, 0])
        z_axis = (0.1 * rot[:, 2] + position, [0, 0, 1])
        if self.x_axis_id is None:
            self.x_axis_id = p.addUserDebugLine(lineFromXYZ, x_axis[0], x_axis[1], lineWidth=3.0)
            self.y_axis_id = p.addUserDebugLine(lineFromXYZ, y_axis[0], y_axis[1], lineWidth=3.0)
            self.z_axis_id = p.addUserDebugLine(lineFromXYZ, z_axis[0], z_axis[1], lineWidth=3.0)
        else:
            self.x_axis_id = p.addUserDebugLine(
                lineFromXYZ, x_axis[0], x_axis[1], lineWidth=3.0, replaceItemUniqueId=self.x_axis_id
            )
            self.y_axis_id = p.addUserDebugLine(
                lineFromXYZ, y_axis[0], y_axis[1], lineWidth=3.0, replaceItemUniqueId=self.y_axis_id
            )
            self.z_axis_id = p.addUserDebugLine(
                lineFromXYZ, z_axis[0], z_axis[1], lineWidth=3.0, replaceItemUniqueId=self.z_axis_id
            )
        # todo: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.i3ffpefe7f3

    @register.inputs(tick=Space(shape=(), dtype="int64"), pos=Space(dtype="float32"), orientation=Space(dtype="float32"))
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, tick: Msg = None, pos: Msg = None, orientation: Msg = None):
        """Produces a camera sensor measurement called `image`.

        If 'position' and 'orientation' were selected as inputs in the spec, the pose of the camera is recalculated before
        rendering the image. Hence, this sensor is able to render images from the perspective of e.g. and end-effector.

        The image measurement is published at the specified rate * real_time_factor.

        Input `tick` ensures that this node is I/O synchronized with the simulator."""
        if pos:
            self.pos = pos.msgs[-1]
        if orientation:
            self.orientation = orientation.msgs[-1]

        if pos is not None or orientation is not None:
            if self.debug:
                self._debug_plot_camera_pose(self.pos, self.orientation, self._p)
            self.cb_args["viewMatrix"] = self._view_matrix(self.pos, self.orientation, self._p._client)
        img = self.cam_cb()
        return dict(image=img)

    @staticmethod
    def _view_matrix(position, orientation, physicsClientId):
        r = R.from_quat(orientation)
        cameraEyePosition = position
        cameraTargetPosition = r.as_matrix()[:, 2] + position  # Assume z axis points outward of the image.
        cameraUpVector = -r.as_matrix()[:, 1]  # Assume y-axis is the up vector in the image
        return pybullet.computeViewMatrix(
            cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=physicsClientId
        )

    @staticmethod
    def _camera_measurement(p, mode, cb_args):
        if mode in ["rgb", "bgr"]:

            def cb():
                if "lightDirection" in cb_args:
                    p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, lightPosition=cb_args["lightDirection"])
                _, _, rgba, depth, seg = p.getCameraImage(**cb_args)
                rgba = rgba if mode == "rgb" else cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                return rgba[:, :, :3]

        elif mode in ["rgba", "bgra"]:

            def cb():
                _, _, rgba, depth, seg = p.getCameraImage(**cb_args)
                rgba = rgba if mode == "rgba" else cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                return rgba[:, :, :4]

        elif mode in ["rgbd", "bgrd"]:

            def cb():
                _, _, rgba, depth, seg = p.getCameraImage(**cb_args)
                rgba = rgba if mode == "rgbd" else cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                depth *= 255  # Convert depth to uint8
                rgba[:, :, 3] = depth.astype("uint8")  # replace alpha channel with depth
                return rgba[:, :, :4]

        else:
            raise ValueError(f"Mode '{mode}' not recognized.")
        return cb
