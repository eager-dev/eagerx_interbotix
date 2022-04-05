# ROS IMPORTS
from urdf_parser_py.urdf import URDF
from std_msgs.msg import Float32MultiArray

# EAGERx IMPORTS
from eagerx_reality.bridge import RealBridge
from eagerx_pybullet.bridge import PybulletBridge
from eagerx import Object, EngineNode, SpaceConverter, EngineState, Processor
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
from eagerx_interbotix.utils import generate_urdf, get_configs


class Xseries(Object):
    entity_id = "Xseries"

    @staticmethod
    @register.sensors(pos=Float32MultiArray, vel=Float32MultiArray)
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
        mode_config=None,
        motor_config=None,
        joint_lower=None,
        joint_upper=None,
        vel_limit=None,
        sleep_positions=None,
        urdf=None,
    )
    def agnostic(spec: ObjectSpec, rate):
        """Agnostic definition of the Xseries"""
        # Register standard converters, space_converters, and processors
        import eagerx.converters  # noqa # pylint: disable=unused-import

        # Set observation properties: (space_converters, rate, etc...)
        spec.sensors.pos.rate = rate
        spec.sensors.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low="$(config joint_lower)",
            high="$(config joint_upper)",
        )
        spec.sensors.vel.rate = rate
        spec.sensors.vel.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=[-v for v in spec.config.vel_limit],
            high=spec.config.vel_limit,
        )

        # Set actuator properties: (space_converters, rate, etc...)
        spec.actuators.pos_control.rate = rate
        spec.actuators.gripper_control.rate = 1
        spec.actuators.pos_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low="$(config joint_lower)",
            high="$(config joint_upper)",
        )
        spec.actuators.gripper_control.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", dtype="float32", low=[0], high=[1.0]
        )

        # Set model_state properties: (space_converters)
        spec.states.pos.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=len(spec.config.joint_names) * [0],
            high=len(spec.config.joint_names) * [0],
        )
        spec.states.vel.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray",
            dtype="float32",
            low=len(spec.config.joint_names) * [0],
            high=len(spec.config.joint_names) * [0],
        )
        spec.states.gripper.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", dtype="float32", low=[0.5], high=[0.5]
        )

    @staticmethod
    @register.spec(entity_id, Object)
    def spec(
        spec: ObjectSpec,
        name: str,
        robot_type: str,
        sensors=None,
        actuators=None,
        states=None,
        rate=30,
        base_pos=None,
        base_or=None,
        self_collision=False,
        fixed_base=True,
        motor_config=None,
        mode_config=None,
    ):
        """Object spec of Xseries"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        Xseries.initialize_spec(spec)

        # Extract info on xseries arm from assets
        motor_config, mode_config = get_configs(robot_type, motor_config, mode_config)
        urdf = URDF.from_parameter_server(generate_urdf(robot_type, ns="pybullet_urdf"))

        # Sort joint_names according to joint_order
        joint_names, sleep_positions = [], []
        for n, sp in zip(motor_config["joint_order"], motor_config["sleep_positions"]):
            if n in motor_config["groups"]["arm"]:
                joint_names.append(n)
                sleep_positions.append(sp)
        motor_config["groups"]["arm"] = joint_names
        joint_names = motor_config["groups"]["arm"]
        gripper_names = [
            motor_config["grippers"]["gripper"]["left_finger"],
            motor_config["grippers"]["gripper"]["right_finger"],
        ]

        # Determine joint limits
        joint_lower, joint_upper, vel_limit = [], [], []
        for n in joint_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == n), None)
            joint_lower.append(joint_obj.limit.lower)
            joint_upper.append(joint_obj.limit.upper)
            vel_limit.append(joint_obj.limit.velocity)

        # Modify default config
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pos"]
        spec.config.actuators = actuators if actuators else ["pos_control", "gripper_control"]
        spec.config.states = states if states else ["pos", "vel", "gripper"]

        # Add registered config params
        spec.config.robot_type = robot_type
        spec.config.joint_names = joint_names
        spec.config.gripper_names = gripper_names
        spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        spec.config.motor_config = motor_config
        spec.config.mode_config = mode_config
        spec.config.joint_lower = joint_lower
        spec.config.joint_upper = joint_upper
        spec.config.vel_limit = vel_limit
        spec.config.sleep_positions = sleep_positions
        spec.config.urdf = urdf.to_xml_string()

        # Add agnostic implementation
        Xseries.agnostic(spec, rate)

    @staticmethod
    # This decorator pre-initializes bridge implementation with default object_params
    @register.bridge(entity_id, PybulletBridge)
    def pybullet_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_interbotix.xseries.pybullet  # noqa # pylint: disable=unused-import
        import eagerx_pybullet  # noqa # pylint: disable=unused-import

        # Set object arguments (as registered per register.bridge_params(..) above the bridge.add_object(...) method.
        spec.PybulletBridge.urdf = generate_urdf(spec.config.robot_type, ns="pybullet_urdf")
        spec.PybulletBridge.basePosition = spec.config.base_pos
        spec.PybulletBridge.baseOrientation = spec.config.base_or
        spec.PybulletBridge.fixed_base = spec.config.fixed_base
        spec.PybulletBridge.self_collision = spec.config.self_collision

        # Determine gripper min/max
        urdf = URDF.from_parameter_server(spec.PybulletBridge.urdf)
        lower, upper = [], []
        for name in spec.config.gripper_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == name), None)
            lower.append(joint_obj.limit.lower)
            upper.append(joint_obj.limit.upper)
        constant = abs(lower[0])
        scale = upper[0] - lower[0]

        # Create engine_states (no agnostic states defined in this case)
        joints = spec.config.joint_names
        spec.PybulletBridge.states.gripper = EngineState.make("PbXseriesGripper", spec.config.gripper_names, constant, scale)
        spec.PybulletBridge.states.pos = EngineState.make("JointState", joints=joints, mode="position")
        spec.PybulletBridge.states.vel = EngineState.make("JointState", joints=joints, mode="velocity")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        pos_sensor = EngineNode.make(
            "JointSensor", "pos_sensor", rate=spec.sensors.pos.rate, process=2, joints=joints, mode="position"
        )
        vel_sensor = EngineNode.make(
            "JointSensor", "vel_sensor", rate=spec.sensors.vel.rate, process=2, joints=joints, mode="velocity"
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        pos_control = EngineNode.make(
            "JointController",
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            process=2,
            joints=joints,
            mode="position_control",
            vel_target=len(joints) * [0.0],
            pos_gain=len(joints) * [0.2],
            vel_gain=len(joints) * [1.5],
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
        gripper.inputs.action.converter = Processor.make("MirrorAction", index=0, constant=constant, scale=scale)

        # Connect all engine nodes
        graph.add([pos_sensor, vel_sensor, pos_control, gripper])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(source=vel_sensor.outputs.obs, sensor="vel")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)

        # Check graph validity (commented out)
        # graph.is_valid(plot=True)

    @staticmethod
    @register.bridge(entity_id, RealBridge)
    def reality_bridge(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (reality) of the object."""
        # Import any object specific entities for this bridge
        import eagerx_interbotix.xseries.real  # noqa # pylint: disable=unused-import

        # Determine gripper min/max
        # Create engine_states (no agnostic states defined in this case)
        # todo: Where do we launch the driver file? In this bridge? How to launch in namespace (based on robot_name + ns)?
        # todo: test with rviz (give as a separate option?)
        spec.RealBridge.states.gripper = EngineState.make("DummyState")
        spec.RealBridge.states.pos = EngineState.make("DummyState")
        spec.RealBridge.states.vel = EngineState.make("DummyState")

        # Create sensor engine nodes
        # Rate=None, but we will connect them to sensors (thus will use the rate set in the agnostic specification)
        joints = spec.config.joint_names
        pos_sensor = EngineNode.make(
            "XseriesSensor", "pos_sensor", rate=spec.sensors.pos.rate, joints=joints, process=2, mode="position"
        )
        vel_sensor = EngineNode.make(
            "XseriesSensor", "vel_sensor", rate=spec.sensors.vel.rate, joints=joints, process=2, mode="velocity"
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        pos_control = EngineNode.make(
            "XseriesArm",
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            joints=joints,
            process=2,
            mode="position_control",
        )
        gripper = EngineNode.make("XseriesGripper", "gripper_control", rate=spec.actuators.gripper_control.rate, process=2)

        # Connect all engine nodes
        graph.add([pos_sensor, vel_sensor, pos_control, gripper])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(source=vel_sensor.outputs.obs, sensor="vel")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)
