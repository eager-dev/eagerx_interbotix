import typing as t
from eagerx_pybullet.engine import PybulletEngine
from eagerx_reality.engine import RealEngine
from eagerx import Object, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Solid(Object):
    @classmethod
    @register.sensors(
        position=Space(shape=(3,), dtype="float32"),
        orientation=Space(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], shape=(4,), dtype="float32"),
        yaw=Space(low=0.0, high=3.14 / 2, shape=(), dtype="float32"),
        robot_view=Space(dtype="uint8"),
    )
    @register.engine_states(
        position=Space(low=[-1, -1, 0], high=[1, 1, 0], dtype="float32"),
        velocity=Space(low=[0, 0, 0], high=[0, 0, 0], dtype="float32"),
        orientation=Space(low=[0, 0, -1, -1], high=[0, 0, 1, 1], dtype="float32"),
        angular_vel=Space(low=[0, 0, 0], high=[0, 0, 0], dtype="float32"),
        lateral_friction=Space(low=0.1, high=0.5, shape=(), dtype="float32"),
        # color=Space(low=[0, 0, 0, 1], high=[1, 1, 1, 1], shape=(4,), dtype="float32")
    )
    def make(
        cls,
        name: str,
        urdf: str,
        rate: int,
        cam_intrinsics: t.Dict,
        sensors: t.List[str] = None,
        states: t.List[str] = None,
        base_pos: t.List[float] = None,
        base_or: t.List[float] = None,
        self_collision: bool = True,
        fixed_base: bool = False,
        cam_translation: t.List[float] = None,
        cam_rotation: t.List[float] = None,
        cam_index: int = 1,
    ) -> ObjectSpec:
        """Object spec of Solid"""
        spec = cls.get_specification()

        # Modify default agnostic params
        spec.config.name = name
        spec.config.sensors = sensors if sensors is not None else ["position", "orientation"]
        spec.config.states = states if states is not None else ["position", "orientation"]

        # Add registered agnostic params
        spec.config.urdf = urdf
        spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.cam_index = cam_index
        spec.config.cam_translation = cam_translation if cam_translation is not None else [1.0, 0.0, 0.2]
        spec.config.cam_rotation = cam_rotation if cam_rotation is not None else [0, 0, 1, 0]
        spec.config.cam_intrinsics = cam_intrinsics

        # Set rates
        spec.sensors.orientation.rate = rate
        spec.sensors.yaw.rate = rate
        spec.sensors.position.rate = rate
        spec.sensors.robot_view.rate = rate

        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.
        spec.engine.urdf = spec.config.urdf
        spec.engine.basePosition = spec.config.base_pos
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = spec.config.fixed_base
        spec.engine.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_pybullet.enginestates import LinkState, PbDynamics

        # from eagerx_interbotix.xseries.pybullet.enginestates import LinkColorState

        spec.engine.states.position = LinkState.make(mode="position")
        spec.engine.states.velocity = LinkState.make(mode="velocity")
        spec.engine.states.orientation = LinkState.make(mode="orientation")
        spec.engine.states.angular_vel = LinkState.make(mode="angular_vel")
        spec.engine.states.lateral_friction = PbDynamics.make(parameter="lateralFriction")
        # spec.engine.states.color = LinkColorState.make()

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor, CameraSensor

        pos = LinkSensor.make("position", rate=spec.sensors.position.rate, mode="position")
        orientation = LinkSensor.make("orientation", rate=spec.sensors.orientation.rate, mode="orientation")

        # Initialize robot view
        height, width = spec.config.cam_intrinsics["image_height"], spec.config.cam_intrinsics["image_width"]
        robot_view = CameraSensor.make(
            "robot_view",
            rate=spec.sensors.robot_view.rate,
            mode="bgr",
            render_shape=[height, width],
            fov=45.0,
            near_val=0.1,
            far_val=10.0,
            debug=True,
        )
        translation = spec.config.cam_translation
        rotation = spec.config.cam_rotation
        robot_view.states.pos.space = Space(low=translation, high=translation)
        robot_view.states.orientation.space = Space(low=rotation, high=rotation)

        # Create wrapped yaw sensor
        from eagerx_interbotix.solid.yaw_node import WrappedYawSensor

        yaw = WrappedYawSensor.make("yaw", rate=spec.sensors.yaw.rate)

        # Connect all engine nodes
        graph.add([pos, orientation, yaw, robot_view])
        graph.connect(source=pos.outputs.obs, sensor="position")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=orientation.outputs.obs, target=yaw.inputs.orientation)
        graph.connect(source=yaw.outputs.yaw, sensor="yaw")
        graph.connect(source=robot_view.outputs.image, sensor="robot_view")

    @staticmethod
    @register.engine(RealEngine)
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Reality) of the object."""
        # Create engine_states (no agnostic states defined in this case)
        from eagerx_interbotix.solid.real.enginestates import DummyState, HumanReset

        spec.engine.states.position = HumanReset.make(description="Waiting for object position reset (w.r.t robot base) to")
        spec.engine.states.velocity = DummyState.make()
        spec.engine.states.orientation = HumanReset.make(
            description="Waiting for object orientation reset (w.r.t robot base) to"
        )
        spec.engine.states.angular_vel = DummyState.make()
        spec.engine.states.lateral_friction = DummyState.make()

        # Ensure all sensors rates are the same (because all sensor measurements come from the same engine node)
        # If they must run at separate rates, split the output of ArucoPoseDetector into different engine nodes.
        rates = [spec.sensors.position.rate, spec.sensors.orientation.rate, spec.sensors.robot_view.rate]
        assert len(set(rates)) == 1, "All sensor rates should be equal."

        # Create engine nodes
        from eagerx_interbotix.solid.real.enginenodes import PoseDetector
        from eagerx_reality.enginenodes import CameraRender

        height, width = spec.config.cam_intrinsics["image_height"], spec.config.cam_intrinsics["image_width"]
        cam = CameraRender.make(
            "cam", spec.sensors.robot_view.rate, camera_idx=spec.config.cam_index, always_render=True, shape=[height, width]
        )
        aruco = PoseDetector.make(
            "aruco",
            spec.sensors.position.rate,
            aruco_id=25,
            aruco_size=0.08,
            aruco_type="DICT_ARUCO_ORIGINAL",
            object_translation=[0, 0, -0.05],
            cam_translation=spec.config.cam_translation,
            cam_rotation=spec.config.cam_rotation,
            cam_intrinsics=spec.config.cam_intrinsics,
        )
        aruco.states.position.space = spec.states.position.space

        # Create wrapped yaw sensor
        from eagerx_interbotix.solid.yaw_node import WrappedYawSensor

        yaw = WrappedYawSensor.make("yaw", rate=spec.sensors.yaw.rate)

        # Connect (see register decorators).
        graph.add([cam, aruco, yaw])
        graph.connect(cam.outputs.image, aruco.inputs.image)
        graph.connect(aruco.outputs.position, sensor="position")
        graph.connect(yaw.outputs.orientation, sensor="orientation")
        graph.connect(aruco.outputs.orientation, target=yaw.inputs.orientation)
        graph.connect(yaw.outputs.yaw, sensor="yaw")
        graph.connect(aruco.outputs.image_aruco, sensor="robot_view")
