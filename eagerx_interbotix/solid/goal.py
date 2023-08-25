import typing as t
from eagerx_pybullet.engine import PybulletEngine
from eagerx_reality.engine import RealEngine
from eagerx import Object, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
from eagerx_interbotix.solid.yaw_node import WrappedYawSensor


class Goal(Object):
    @classmethod
    @register.sensors(
        position=Space(shape=(3,), dtype="float32"),
        orientation=Space(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], shape=(4,), dtype="float32"),
        yaw=Space(low=0.0, high=3.14 / 2, shape=(), dtype="float32"),
    )
    @register.engine_states(
        position=Space(low=[-1, -1, 0], high=[1, 1, 0], dtype="float32"),
        orientation=Space(low=[0, 0, -1, -1], high=[0, 0, 1, 1], dtype="float32"),
        # color=Space(low=[0, 0, 0, 1], high=[1, 1, 1, 1], dtype="float32"),
    )
    def make(
        cls,
        name: str,
        urdf: str,
        rate: int,
        sensors: t.List[str] = None,
        states: t.List[str] = None,
        base_pos: t.List[float] = None,
        base_or: t.List[float] = None,
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

        # Set rates
        spec.sensors.orientation.rate = rate
        spec.sensors.yaw.rate = rate
        spec.sensors.position.rate = rate
        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.
        spec.engine.urdf = spec.config.urdf
        spec.engine.basePosition = spec.config.base_pos
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = True
        spec.engine.self_collision = False

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_pybullet.enginestates import LinkState

        # from eagerx_interbotix.solid.pybullet.enginestates import TextureState
        # from eagerx_interbotix.xseries.pybullet.enginestates import LinkColorState

        spec.engine.states.position = LinkState.make(mode="position")
        spec.engine.states.orientation = LinkState.make(mode="orientation")
        # spec.engine.states.color = LinkColorState.make()
        # texture_path = str(Path(__file__).parent / "assets" / "rect2.png")
        # spec.engine.states.color = TextureState.make(texture_path=texture_path)

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor

        pos = LinkSensor.make("position", rate=spec.sensors.position.rate, mode="position")
        orientation = LinkSensor.make("orientation", rate=spec.sensors.orientation.rate, mode="orientation")
        yaw = WrappedYawSensor.make("yaw", rate=spec.sensors.yaw.rate)

        # Connect all engine nodes
        graph.add([pos, orientation, yaw])
        graph.connect(source=pos.outputs.obs, sensor="position")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=orientation.outputs.obs, target=yaw.inputs.orientation)
        graph.connect(source=yaw.outputs.yaw, sensor="yaw")

    @staticmethod
    @register.engine(RealEngine)
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Reality) of the object."""
        # Create engine_states (no agnostic states defined in this case)
        from eagerx_interbotix.solid.real.enginestates import GoalState

        spec.engine.states.position = GoalState.make(mode="position")
        spec.engine.states.orientation = GoalState.make(mode="orientation")

        # Create sensor engine nodes
        from eagerx_interbotix.solid.real.enginenodes import GoalObservationSensor, YawGoalObservationSensor

        pos = GoalObservationSensor.make("position", rate=spec.sensors.position.rate, mode="position")
        orientation = GoalObservationSensor.make("orientation", rate=spec.sensors.orientation.rate, mode="orientation")
        yaw = YawGoalObservationSensor.make("yaw", rate=spec.sensors.yaw.rate)

        # Connect all engine nodes
        graph.add([pos, orientation, yaw])
        graph.connect(source=pos.outputs.obs, sensor="position")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=yaw.outputs.obs, sensor="yaw")
