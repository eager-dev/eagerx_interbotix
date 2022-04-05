# ROS IMPORTS
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

# EAGERx IMPORTS
from eagerx_pybullet.bridge import PybulletBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Camera(Object):
    entity_id = "Camera"

    @staticmethod
    @register.sensors(rgb=Image, pos=Float32MultiArray, orientation=Float32MultiArray)
    @register.engine_states(pos=Float32MultiArray, orientation=Float32MultiArray)
    @register.config(
        urdf=None,
        fixed_base=True,
        self_collision=True,
        base_pos=[0, 0, 0],
        base_or=[0, 0, 0, 1],
        render_shape=[480, 640],
        optical_link=None,
        calibration_link=None,
    )
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the Camera"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Position
        spec.sensors.pos.rate = rate
        spec.sensors.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-999, -999, -999],
            high=[999, 999, 999],
        )
        spec.states.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0.83, 0.0181, 0.75],
            high=[0.83, 0.0181, 0.75],
        )

        # Orientation
        spec.sensors.orientation.rate = rate
        spec.sensors.orientation.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-1, -1, -1, -1],
            high=[1, 1, 1, 1],
        )
        spec.states.orientation.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[0.377, -0.04, -0.92, 0.088],
            high=[0.377, -0.04, -0.92, 0.088],
        )

        # Rgb
        spec.sensors.rgb.rate = rate
        spec.sensors.rgb.space_converter = SpaceConverter.make(
            "Space_Image",
            dtype="float32",
            low=0,
            high=1,
            shape=spec.config.render_shape + [3],
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
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
    ):
        """Object spec of Camera"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Camera.initialize_spec(spec)

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

        # Add agnostic implementation
        Camera.agnostic(spec, rate)

    @staticmethod
    @register.bridge(entity_id, PybulletBridge)
    def pybullet_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_interbotix.solid.pybullet  # noqa # pylint: disable=unused-import
        import eagerx_pybullet  # noqa # pylint: disable=unused-import

        # Set object arguments (as registered per register.bridge_params(..) above the bridge.add_object(...) method.
        import pybullet_data

        urdf = spec.config.urdf
        spec.PybulletBridge.urdf = (
            urdf if isinstance(urdf, str) else "%s/%s.urdf" % (pybullet_data.getDataPath(), "cube_small")
        )
        # Initial position of baselink when urdf is loaded. Overwritten by state during the reset.
        spec.PybulletBridge.basePosition = spec.config.base_pos
        # Initial orientation of baselink when urdf is loaded. Overwritten by state during the reset.
        spec.PybulletBridge.baseOrientation = spec.config.base_or
        spec.PybulletBridge.fixed_base = spec.config.fixed_base
        spec.PybulletBridge.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        spec.PybulletBridge.states.pos = EngineState.make("LinkState", mode="position", link=spec.config.calibration_link)
        spec.PybulletBridge.states.orientation = EngineState.make(
            "LinkState", mode="orientation", link=spec.config.calibration_link
        )

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        pos = EngineNode.make(
            "LinkSensor", "pos", rate=spec.sensors.pos.rate, process=2, mode="position", links=[spec.config.optical_link]
        )
        orientation = EngineNode.make(
            "LinkSensor",
            "orientation",
            rate=spec.sensors.orientation.rate,
            process=2,
            mode="orientation",
            links=[spec.config.optical_link],
        )
        rgb = EngineNode.make(
            "CameraSensor", "rgb", rate=spec.sensors.rgb.rate, process=2, mode="rgb", render_shape=spec.config.render_shape
        )
        rgb.config.inputs.append("pos")
        rgb.config.inputs.append("orientation")

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        # Connect all engine nodes
        graph.add([pos, orientation, rgb])
        graph.connect(source=pos.outputs.obs, sensor="pos")
        graph.connect(source=orientation.outputs.obs, sensor="orientation")
        graph.connect(source=rgb.outputs.image, sensor="rgb")
        graph.connect(source=pos.outputs.obs, target=rgb.inputs.pos)
        graph.connect(source=orientation.outputs.obs, target=rgb.inputs.orientation)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
