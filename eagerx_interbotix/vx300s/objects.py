# ROS IMPORTS
from std_msgs.msg import Float32MultiArray

# EAGERx IMPORTS
from eagerx_pybullet.bridge import PybulletBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState, Processor
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register

# OTHER IMPORTS
import os


class Vx300s(Object):
    entity_id = "Vx300s"

    @staticmethod
    @register.sensors(pos=Float32MultiArray)
    @register.actuators(pos_control=Float32MultiArray, gripper_control=Float32MultiArray)
    @register.engine_states(pos=Float32MultiArray, vel=Float32MultiArray, gripper=Float32MultiArray)
    @register.config(
        robot_type=None,
        joint_names=None,
        gripper_names=None,
        fixed_base=True,
        self_collision=True,
        base_pos=None,
        base_or=None,
    )
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the Vx300s"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.pos.rate = rate
        spec.sensors.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159],
            high=[3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159],
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.pos_control.rate = rate
        spec.actuators.gripper_control.rate = rate
        spec.actuators.pos_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159],
            high=[3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159],
        )
        spec.actuators.gripper_control.converter = Processor.make("MirrorAction", index=0, offset=0)
        spec.actuators.gripper_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", dtype="float32", low=[0.021], high=[0.057]
        )

        # Set model_state properties: (space_converters)
        spec.states.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-3.14158, -1.85004, -1.76278, -3.14158, -1.86750, -3.14158],
            high=[3.14158, 1.25663, 1.605702, 3.14158, 2.23402, 3.14158],
        )
        spec.states.vel.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", dtype="float32", low=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        spec.states.gripper.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", dtype="float32", low=[0.021, -0.057], high=[0.057, -0.021]
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        sensors=None,
        actuators=None,
        states=None,
        rate=30,
        base_pos=None,
        base_or=None,
        self_collision=True,
        fixed_base=True,
    ):
        """Object spec of Vx300s"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Vx300s.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pos"]
        spec.config.actuators = actuators if actuators else ["pos_control", "gripper_control"]
        spec.config.states = states if states else ["pos", "vel", "gripper"]

        # Add registered agnostic params
        spec.config.robot_type = "vx300s"
        spec.config.joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        spec.config.gripper_names = ["left_finger", "right_finger"]
        spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base

        # Add agnostic implementation
        Vx300s.agnostic(spec, rate)

    @staticmethod
    @register.bridge(
        entity_id, PybulletBridge
    )  # This decorator pre-initializes bridge implementation with default object_params
    def pybullet_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_interbotix.vx300s.pybullet  # noqa # pylint: disable=unused-import
        import eagerx_pybullet  # noqa # pylint: disable=unused-import

        # Set object arguments (as registered per register.bridge_params(..) above the bridge.add_object(...) method.
        path = os.path.dirname(eagerx_interbotix.__file__)
        path += f"/../descriptions/urdf/{spec.config.robot_type}.urdf"
        spec.PybulletBridge.urdf = path
        spec.PybulletBridge.basePosition = spec.config.base_pos
        spec.PybulletBridge.baseOrientation = spec.config.base_or
        spec.PybulletBridge.fixed_base = spec.config.fixed_base
        spec.PybulletBridge.self_collision = spec.config.self_collision

        # Create engine_states (no agnostic states defined in this case)
        spec.PybulletBridge.states.pos = EngineState.make("JointState", joints=spec.config.joint_names, mode="position")
        spec.PybulletBridge.states.vel = EngineState.make("JointState", joints=spec.config.joint_names, mode="velocity")
        spec.PybulletBridge.states.gripper = EngineState.make("JointState", joints=spec.config.gripper_names, mode="position")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        pos_sensor = EngineNode.make(
            "JointSensor", "pos_sensor", rate=spec.sensors.pos.rate, process=2, joints=spec.config.joint_names, mode="position"
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        pos_control = EngineNode.make(
            "JointController",
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            process=2,
            joints=spec.config.joint_names,
            mode="position_control",
            vel_target=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pos_gain=[0.45, 0.45, 0.65, 0.6, 0.45, 0.4],
            vel_gain=[1.7, 1.7, 1.5, 1.3, 1.0, 1.0],
        )
        gripper = EngineNode.make(
            "JointController",
            "gripper_control",
            rate=spec.actuators.gripper_control.rate,
            process=2,
            joints=spec.config.gripper_names,
            mode="position_control",
            vel_target=[0.0, 0.0],
            pos_gain=[1.5, 1.5],
            vel_gain=[0.7, 0.7],
        )

        # Connect all engine nodes
        graph.add([pos_sensor, pos_control, gripper])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
