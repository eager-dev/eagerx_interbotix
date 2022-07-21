import eagerx
from eagerx import Space
from eagerx_reality.engine import RealEngine
from eagerx_pybullet.engine import PybulletEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
from eagerx_interbotix.utils import generate_urdf, get_configs

# ROS IMPORTS
from urdf_parser_py.urdf import URDF


class Xseries(eagerx.Object):
    @classmethod
    @register.sensors(
        pos=Space(dtype="float32"), vel=Space(dtype="float32"), ee_pos=Space(low=[-2, -2, 0], high=[2, 2, 2], dtype="float32")
    )
    @register.actuators(
        pos_control=Space(dtype="float32"),
        vel_control=Space(dtype="float32"),
        gripper_control=Space(low=[0], high=[1], dtype="float32"),
    )
    @register.engine_states(
        pos=Space(dtype="float32"),
        vel=Space(dtype="float32"),
        gripper=Space(low=[0.5], high=[0.5], dtype="float32"),
    )
    def make(
        cls,
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
    ) -> ObjectSpec:
        """Object spec of Xseries"""
        spec = cls.get_specification()

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
        gripper_link = [link.name for link in urdf.links if "ee_gripper_link" in link.name][0]

        # Determine joint limits
        joint_lower, joint_upper, vel_limit = [], [], []
        for n in joint_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == n), None)
            joint_lower.append(joint_obj.limit.lower)
            joint_upper.append(joint_obj.limit.upper)
            vel_limit.append(joint_obj.limit.velocity)

        # Determine gripper limits
        gripper_lower, gripper_upper = [], []
        for name in gripper_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == name), None)
            gripper_lower.append(joint_obj.limit.lower)
            gripper_upper.append(joint_obj.limit.upper)

        # Modify default config
        spec.config.name = name
        spec.config.sensors = sensors if sensors else ["pos"]
        spec.config.actuators = actuators if actuators else ["pos_control", "gripper_control"]
        spec.config.states = states if states else ["pos", "vel", "gripper"]

        # Add registered config params
        spec.config.robot_type = robot_type
        spec.config.joint_names = joint_names
        spec.config.gripper_names = gripper_names
        spec.config.gripper_link = gripper_link
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

        # Set rates
        spec.sensors.pos.rate = rate
        spec.sensors.vel.rate = rate
        spec.sensors.ee_pos.rate = rate
        spec.actuators.pos_control.rate = rate
        spec.actuators.vel_control.rate = rate
        spec.actuators.gripper_control.rate = 1

        # Set variable spaces
        spec.sensors.pos.space.update(low=joint_lower, high=joint_upper)
        spec.sensors.vel.space.update(low=[-v for v in vel_limit], high=vel_limit)
        spec.actuators.pos_control.space.update(low=joint_lower, high=joint_upper)
        spec.actuators.vel_control.space.update(low=[-v for v in vel_limit], high=vel_limit)
        spec.states.pos.space.update(low=[0.0 for _j in joint_lower], high=[0.0 for _j in joint_upper])
        spec.states.vel.space.update(low=[0.0 for _j in joint_lower], high=[0.0 for _j in joint_upper])
        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.)
        spec.engine.urdf = spec.config.urdf
        spec.engine.basePosition = spec.config.base_pos
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = spec.config.fixed_base
        spec.engine.self_collision = spec.config.self_collision

        # Determine gripper min/max
        urdf = URDF.from_parameter_server(generate_urdf(spec.config.robot_type, ns="pybullet_urdf"))
        lower, upper = [], []
        for name in spec.config.gripper_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == name), None)
            lower.append(joint_obj.limit.lower)
            upper.append(joint_obj.limit.upper)
        constant = abs(lower[0])
        scale = upper[0] - lower[0]

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_interbotix.xseries.pybullet.enginestates import PbXseriesGripper
        from eagerx_pybullet.enginestates import JointState

        joints = spec.config.joint_names
        spec.engine.states.gripper = PbXseriesGripper.make(spec.config.gripper_names, constant, scale)
        spec.engine.states.pos = JointState.make(joints=joints, mode="position")
        spec.engine.states.vel = JointState.make(joints=joints, mode="velocity")

        # Fix gripper if we are not controlling it.
        if "gripper_control" not in spec.config.actuators:
            spec.engine.states.gripper.fixed = True

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor, JointSensor, JointController

        pos_sensor = JointSensor.make("pos_sensor", rate=spec.sensors.pos.rate, process=2, joints=joints, mode="position")
        vel_sensor = JointSensor.make("vel_sensor", rate=spec.sensors.vel.rate, process=2, joints=joints, mode="velocity")
        ee_pos_sensor = LinkSensor.make(
            "ee_pos_sensor",
            rate=spec.sensors.ee_pos.rate,
            process=2,
            links=[spec.config.gripper_link],
            mode="position",
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        pos_control = JointController.make(
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            process=2,
            joints=joints,
            mode="position_control",
            vel_target=len(joints) * [0.0],
            pos_gain=len(joints) * [0.5],
            vel_gain=len(joints) * [1.0],
            max_force=len(joints) * [2.0],
        )
        vel_control = JointController.make(
            "vel_control",
            rate=spec.actuators.pos_control.rate,
            process=2,
            joints=joints,
            mode="velocity_control",
            vel_gain=len(joints) * [1.0],
            max_force=len(joints) * [2.0],
        )
        gripper = JointController.make(
            "gripper_control",
            rate=spec.actuators.gripper_control.rate,
            process=2,
            joints=spec.config.gripper_names,
            mode="position_control",
            vel_target=[0.0, 0.0],
            pos_gain=[0.5, 0.5],
            vel_gain=[1.0, 1.0],
            max_force=[2.0, 2.0],
        )
        from eagerx_interbotix.xseries.processor import MirrorAction

        gripper.inputs.action.processor = MirrorAction.make(index=0, constant=constant, scale=scale)

        # Connect all engine nodes
        graph.add([pos_sensor, vel_sensor, ee_pos_sensor, pos_control, vel_control, gripper])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(source=vel_sensor.outputs.obs, sensor="vel")
        graph.connect(source=ee_pos_sensor.outputs.obs, sensor="ee_pos")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="vel_control", target=vel_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)

    @staticmethod
    @register.engine(RealEngine)
    def reality_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (reality) of the object."""
        # Import any object specific entities for this engine
        import eagerx_interbotix.xseries.real  # noqa # pylint: disable=unused-import

        # Determine gripper min/max
        from eagerx_interbotix.xseries.real.enginestates import DummyState

        spec.engine.states.gripper = DummyState.make()
        spec.engine.states.pos = DummyState.make()
        spec.engine.states.vel = DummyState.make()

        # Create sensor engine nodes
        from eagerx_interbotix.xseries.real.enginenodes import XseriesGripper, XseriesSensor, XseriesArm

        joints = spec.config.joint_names
        pos_sensor = XseriesSensor.make("pos_sensor", rate=spec.sensors.pos.rate, joints=joints, process=2, mode="position")
        vel_sensor = XseriesSensor.make("vel_sensor", rate=spec.sensors.vel.rate, joints=joints, process=2, mode="velocity")

        # Create actuator engine nodes
        pos_control = XseriesArm.make(
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            joints=joints,
            process=2,
            mode="position_control",
        )
        gripper = XseriesGripper.make("gripper_control", rate=spec.actuators.gripper_control.rate, process=2)

        # Connect all engine nodes
        graph.add([pos_sensor, vel_sensor, pos_control, gripper])
        graph.connect(source=pos_sensor.outputs.obs, sensor="pos")
        graph.connect(source=vel_sensor.outputs.obs, sensor="vel")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)
