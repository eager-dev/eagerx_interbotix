from gym.spaces import Box
import numpy as np
import eagerx
from eagerx_pybullet.engine import PybulletEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Camera(eagerx.Object):
    @classmethod
    @register.sensors(
        pos=Box(
            low=np.array([-999, -999, -999], dtype="float32"),
            high=np.array([999, 999, 999], dtype="float32"),
        ),
        orientation=Box(
            low=np.array([-1, -1, -1, -1], dtype="float32"),
            high=np.array([1, 1, 1, 1], dtype="float32"),
        ),
        rgb=None,
    )
    @register.engine_states(
        pos=Box(
            low=np.array([0.83, 0.0181, 0.75], dtype="float32"),
            high=np.array([0.83, 0.0181, 0.75], dtype="float32"),
        ),
        orientation=Box(
            low=np.array([0.377, -0.04, -0.92, 0.088], dtype="float32"),
            high=np.array([0.377, -0.04, -0.92, 0.088], dtype="float32"),
        ),
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
    ) -> ObjectSpec:
        """Object spec of Camera"""
        spec = cls.get_specification()

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if isinstance(sensors, list) else ["rgb"]
        spec.config.states = states if isinstance(states, list) else ["pos", "orientation"]

        # Add registered agnostic params
        spec.config.urdf = urdf
        spec.config.base_pos = base_pos if isinstance(base_pos, list) else [0, 0, 0]
        spec.config.base_or = base_or if isinstance(base_or, list) else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.render_shape = render_shape if isinstance(render_shape, list) else [200, 300]
        spec.config.optical_link = optical_link if isinstance(optical_link, str) else None
        spec.config.calibration_link = calibration_link if isinstance(calibration_link, str) else None

        # Set rates
        spec.sensors.rgb.rate = rate
        spec.sensors.pos.rate = rate
        spec.sensors.orientation.rate = rate

        # Set variable space limits
        spec.sensors.rgb.space = Box(
            dtype="uint8",
            low=0,
            high=255,
            shape=tuple(spec.config.render_shape + [3]),
        )

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
        from eagerx_pybullet.enginenodes import LinkSensor, CameraSensor

        pos = LinkSensor.make("pos", rate=spec.sensors.pos.rate, process=2, mode="position", links=[spec.config.optical_link])
        orientation = LinkSensor.make(
            "orientation",
            rate=spec.sensors.orientation.rate,
            process=2,
            mode="orientation",
            links=[spec.config.optical_link],
        )
        rgb = CameraSensor.make(
            "rgb", rate=spec.sensors.rgb.rate, process=2, mode="rgb", render_shape=spec.config.render_shape
        )
        rgb.config.inputs.append("pos")
        rgb.config.inputs.append("orientation")

        # Connect all engine nodes
        graph.add([pos, orientation, rgb])
        graph.connect(source=pos.outputs.obs, sensor="pos")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=rgb.outputs.image, sensor="rgb")
        graph.connect(source=pos.outputs.obs, target=rgb.inputs.pos)
        graph.connect(source=orientation.outputs.obs, target=rgb.inputs.orientation)
