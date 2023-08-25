import eagerx
from eagerx import Space
from eagerx_pybullet.engine import PybulletEngine
from eagerx_reality.engine import RealEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Camera(eagerx.Object):
    @classmethod
    @register.sensors(
        pos=Space(shape=(3,), dtype="float32"),
        orientation=Space(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], shape=(4,), dtype="float32"),
        image=Space(dtype="uint8"),
    )
    @register.engine_states(
        pos=Space(low=[0.83, 0.0181, 0.75], high=[0.83, 0.0181, 0.75], shape=(3,), dtype="float32"),
        orientation=Space(low=[0.377, -0.04, -0.92, 0.088], high=[0.377, -0.04, -0.92, 0.088], shape=(4,), dtype="float32"),
    )
    def make(
        cls,
        name: str,
        sensors=None,
        states=None,
        rate=30,
        base_pos=None,
        base_or=None,
        self_collision=True,
        fixed_base=True,
        render_shape=None,
        urdf: str = None,
        optical_link: str = None,
        calibration_link: str = None,
        camera_index: int = 0,
        mode: str = "bgr",
        fov: float = 45.0,
        near_val: float = 0.1,
        far_val: float = 10.0,
        light_direction_low=None,
        light_direction_high=None,
    ) -> ObjectSpec:
        """Make a spec to initialize a camera.

        :param name: Name of the object (topics are placed within this namespace).
        :param sensors: A list of selected sensors. Must be a subset of the registered sensors.
        :param states: A list of selected states. Must be a subset of the registered actuators.
        :param rate: The default rate at which all sensors run. Can be modified via the spec API.
        :param base_pos: Base position of the object [x, y, z].
        :param base_or: Base orientation of the object in quaternion [x, y, z, w].
        :param self_collision: Enable self collisions.
        :param fixed_base: Force the base of the loaded object to be static.
        :param render_shape: The shape of the produced images [height, width].
        :param urdf: A fullpath (ending with .urdf), a key that points to the urdf (xml)string on the
                     rosparam server, or a urdf within pybullet's search path. The `pybullet_data` package is
                     included in the search path.
        :param optical_link: Link related to the pose from which to render images.
        :param calibration_link: Link related to the pose that is reset.
        :param camera_index: Camera index corresponding to the camera device number per OpenCV.
        :param mode: Available: `rgb`, `bgr`.
        :param fov: Field of view.
        :param near_val: Near plane distance [m].
        :param far_val: Far plane distance [m].
        :param light_direction: Specifies the world position of the light source, the direction is from the light source
                                position to the origin of the world frame., list of 3 floats.
        :return: ObjectSpec
        """
        spec = cls.get_specification()

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if isinstance(sensors, list) else ["image"]
        spec.config.states = states if isinstance(states, list) else ["pos", "orientation"]

        # Add registered agnostic params
        spec.config.urdf = urdf
        spec.config.base_pos = base_pos if isinstance(base_pos, list) else [0, 0, 0]
        spec.config.base_or = base_or if isinstance(base_or, list) else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.render_shape = render_shape if isinstance(render_shape, list) else [480, 480]
        spec.config.optical_link = optical_link if isinstance(optical_link, str) else None
        spec.config.calibration_link = calibration_link if isinstance(calibration_link, str) else None
        spec.config.camera_index = camera_index
        spec.config.fov = fov
        spec.config.near_val = near_val
        spec.config.far_val = far_val
        spec.config.light_direction_low = light_direction_low
        spec.config.light_direction_high = light_direction_high

        # Set rates
        spec.sensors.image.rate = rate
        spec.sensors.pos.rate = rate
        spec.sensors.orientation.rate = rate

        # Set variable space limits
        spec.sensors.image.space.update(low=0, high=255, shape=list(spec.config.render_shape + [3]))
        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.
        import pybullet_data

        urdf = spec.config.urdf
        spec.engine.urdf = urdf if isinstance(urdf, str) else "%s/%s.urdf" % (pybullet_data.getDataPath(), "cube_small")
        # Initial position of baselink when urdf is loaded. Overwritten by state during the reset.
        spec.engine.basePosition = spec.config.base_pos
        # Initial orientation of baselink when urdf is loaded. Overwritten by state during the reset.
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = spec.config.fixed_base
        spec.engine.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_pybullet.enginestates import LinkState

        spec.engine.states.pos = LinkState.make(mode="position", link=spec.config.calibration_link)
        spec.engine.states.orientation = LinkState.make(mode="orientation", link=spec.config.calibration_link)

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor
        from eagerx_interbotix.camera.pybullet.enginenodes import CameraSensor

        pos = LinkSensor.make("pos", rate=spec.sensors.pos.rate, process=2, mode="position", links=[spec.config.optical_link])
        orientation = LinkSensor.make(
            "orientation",
            rate=spec.sensors.orientation.rate,
            process=2,
            mode="orientation",
            links=[spec.config.optical_link],
        )
        image = CameraSensor.make(
            "image",
            rate=spec.sensors.image.rate,
            inputs=["pos", "orientation"],
            process=2,
            mode="bgr",
            render_shape=spec.config.render_shape,
            fov=spec.config.fov,
            near_val=spec.config.near_val,
            far_val=spec.config.far_val,
            states=["light_direction"],
            debug=True,
        )
        if spec.config.light_direction_low:
            image.states.light_direction.space.update(low=spec.config.light_direction_low)
        if spec.config.light_direction_high:
            image.states.light_direction.space.update(high=spec.config.light_direction_high)

        # Connect all engine nodes
        graph.add([pos, orientation, image])
        graph.connect(source=pos.outputs.obs, sensor="pos")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=image.outputs.image, sensor="image")
        graph.connect(source=pos.outputs.obs, target=image.inputs.pos)
        graph.connect(source=orientation.outputs.obs, target=image.inputs.orientation)

    @staticmethod
    @register.engine(RealEngine)
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Couple engine states
        from eagerx_interbotix.camera.enginestates import DummyState

        spec.engine.states.pos = DummyState.make()
        spec.engine.states.orientation = DummyState.make()

        # Create sensor engine nodes
        from eagerx_reality.enginenodes import CameraRender

        image = CameraRender.make(
            "image",
            shape=spec.config.render_shape,
            rate=spec.sensors.image.rate,
            process=eagerx.process.NEW_PROCESS,
            camera_idx=spec.config.camera_index,
        )
        graph.add([image])
        graph.connect(source=image.outputs.image, sensor="image")
