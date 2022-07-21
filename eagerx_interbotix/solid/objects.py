from eagerx_pybullet.engine import PybulletEngine
from eagerx import Object, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Solid(Object):
    @classmethod
    @register.sensors(
        pos=Space(shape=(3,), dtype="float32"),
        vel=Space(shape=(3,), dtype="float32"),
        orientation=Space(low=[-1, -1, -1, -1], high=[1, 1, 1, 1], shape=(4,), dtype="float32"),
        angular_vel=Space(shape=(3,), dtype="float32"),
    )
    @register.engine_states(
        pos=Space(low=[-1, -1, 0], high=[1, 1, 0], dtype="float32"),
        vel=Space(low=[0, 0, 0], high=[0, 0, 0], dtype="float32"),
        orientation=Space(low=[0, 0, -1, -1], high=[0, 0, 1, 1], dtype="float32"),
        angular_vel=Space(low=[0, 0, 0], high=[0, 0, 0], dtype="float32"),
        lateral_friction=Space(low=0.1, high=0.5, shape=(), dtype="float32"),
    )
    def make(
        cls,
        name: str,
        urdf: str,
        sensors=None,
        states=None,
        rate=30,
        base_pos=None,
        base_or=None,
        self_collision=True,
        fixed_base=True,
    ) -> ObjectSpec:
        """Object spec of Solid"""
        spec = cls.get_specification()

        # Modify default agnostic params
        spec.config.name = name
        spec.config.sensors = sensors if sensors is not None else ["pos", "vel", "orientation", "angular_vel"]
        spec.config.states = states if states is not None else ["pos", "vel", "orientation", "angular_vel"]

        # Add registered agnostic params
        spec.config.urdf = urdf
        spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base

        # Set rates
        spec.sensors.angular_vel.rate = rate
        spec.sensors.orientation.rate = rate
        spec.sensors.pos.rate = rate
        spec.sensors.vel.rate = rate

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

        spec.engine.states.pos = LinkState.make(mode="position")
        spec.engine.states.vel = LinkState.make(mode="velocity")
        spec.engine.states.orientation = LinkState.make(mode="orientation")
        spec.engine.states.angular_vel = LinkState.make(mode="angular_vel")
        spec.engine.states.lateral_friction = PbDynamics.make(parameter="lateralFriction")

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor

        pos = LinkSensor.make("pos", rate=spec.sensors.pos.rate, process=2, mode="position")
        vel = LinkSensor.make("vel", rate=spec.sensors.vel.rate, process=2, mode="velocity")
        orientation = LinkSensor.make("orientation", rate=spec.sensors.orientation.rate, process=2, mode="orientation")
        angular_vel = LinkSensor.make("angular_vel", rate=spec.sensors.angular_vel.rate, process=2, mode="angular_vel")

        # Create actuator engine nodes
        # Connect all engine nodes
        graph.add([pos, vel, orientation, angular_vel])
        graph.connect(source=pos.outputs.obs, sensor="pos")
        graph.connect(source=vel.outputs.obs, sensor="vel")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=angular_vel.outputs.obs, sensor="angular_vel")
