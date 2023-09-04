import eagerx
from eagerx import Space
from eagerx_reality.engine import RealEngine
from eagerx_pybullet.engine import PybulletEngine
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
from eagerx_interbotix.utils import generate_urdf, get_configs, XmlListConfig

import os
from xml.etree import cElementTree as ElementTree

# ROS IMPORTS
try:
    from urdf_parser_py.urdf import URDF
except ImportError:
    print("Cannot find ROS.")
    URDF = None


class Xseries(eagerx.Object):
    @classmethod
    @register.sensors(
        position=Space(dtype="float32"),
        velocity=Space(dtype="float32"),
        force_torque=Space(low=-20, high=20, shape=(6,), dtype="float32"),
        gripper_position=Space(dtype="float32"),
        ee_pos=Space(low=[-2, -2, 0], high=[2, 2, 2], dtype="float32"),
        ee_orn=Space(low=-1, high=1, shape=(4,), dtype="float32"),
        moveit_status=Space(low=0, high=1, shape=(), dtype="int64"),
    )
    @register.actuators(
        pos_control=Space(dtype="float32"),
        vel_control=Space(dtype="float32"),
        gripper_control=Space(low=[0.0], high=[1.0], dtype="float32"),
        moveit_to=Space(dtype="float32"),
    )
    @register.engine_states(
        position=Space(dtype="float32"),
        velocity=Space(dtype="float32"),
        gripper=Space(low=[0.5], high=[0.5], dtype="float32"),
        color=Space(low=[0.25, 0.25, 0.25, 1], high=[0.25, 0.25, 0.25, 1], shape=(4,), dtype="float32"),
    )
    def make(
        cls,
        name: str,
        robot_type: str,
        arm_name=None,
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
        pressure: float = 0.5,
    ) -> ObjectSpec:
        """Object spec of Xseries"""
        if URDF is None:
            raise ImportError("Ros not installed. Required for generating urdf.")

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
        gripper_lower, gripper_upper, gripper_vel_limit = [], [], []
        for n in gripper_names:
            joint_obj = next((joint for joint in urdf.joints if joint.name == n), None)
            gripper_lower.append(joint_obj.limit.lower)
            gripper_upper.append(joint_obj.limit.upper)
            gripper_vel_limit.append(joint_obj.limit.velocity)

        # Modify default config
        spec.config.name = name
        spec.config.sensors = sensors if isinstance(sensors, list) else ["position"]
        spec.config.actuators = actuators if isinstance(actuators, list) else ["pos_control", "gripper_control"]
        spec.config.states = states if isinstance(states, list) else ["position", "velocity", "gripper", "color"]

        # Add registered config params
        spec.config.robot_type = robot_type
        spec.config.arm_name = arm_name if arm_name else robot_type
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
        spec.config.pressure = pressure

        # Set rates
        spec.sensors.position.rate = rate
        spec.sensors.velocity.rate = rate
        spec.sensors.force_torque.rate = rate
        spec.sensors.gripper_position.rate = rate
        spec.sensors.ee_pos.rate = rate
        spec.sensors.ee_orn.rate = rate
        spec.sensors.moveit_status.rate = rate
        spec.actuators.pos_control.rate = rate
        spec.actuators.moveit_to.rate = rate
        spec.actuators.vel_control.rate = rate
        spec.actuators.gripper_control.rate = 1

        # Set variable spaces
        spec.sensors.position.space.update(low=joint_lower, high=joint_upper)
        spec.sensors.velocity.space.update(low=[-v for v in vel_limit], high=vel_limit)
        # spec.sensors.gripper_position.space.update(low=[gripper_lower[0], -gripper_vel_limit[0]], high=[gripper_upper[1], gripper_vel_limit[0]])
        spec.sensors.gripper_position.space.update(low=[gripper_lower[0] * 0.9], high=[gripper_upper[0] * 1.1])
        spec.actuators.pos_control.space.update(low=joint_lower, high=joint_upper)
        spec.actuators.moveit_to.space.update(low=joint_lower, high=joint_upper)
        spec.actuators.vel_control.space.update(low=[-v for v in vel_limit], high=vel_limit)
        spec.states.position.space.update(low=[0.0 for _j in joint_lower], high=[0.0 for _j in joint_upper])
        spec.states.velocity.space.update(low=[0.0 for _j in joint_lower], high=[0.0 for _j in joint_upper])
        return spec

    @staticmethod
    @register.engine(PybulletEngine)
    def pybullet_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (Pybullet) of the object."""
        # Set object arguments (as registered per register.engine_params(..) above the engine.add_object(...) method.)
        module_path = os.path.dirname(__file__) + "/assets/"
        assert os.path.exists(module_path), "Module path does not exist: {}".format(module_path)
        urdf = spec.config.urdf
        urdf_sbtd = urdf.replace("package://interbotix_xsarm_descriptions/", module_path)
        spec.engine.urdf = urdf_sbtd
        spec.engine.basePosition = spec.config.base_pos
        spec.engine.baseOrientation = spec.config.base_or
        spec.engine.fixed_base = spec.config.fixed_base
        spec.engine.self_collision = spec.config.self_collision

        # Determine gripper min/max
        tree = ElementTree.ElementTree(ElementTree.fromstring(urdf_sbtd))
        root = tree.getroot()
        urdf_list = XmlListConfig(root)
        # todo: urdf = URDF.from_xml(spec.config.urdf)
        lower, upper = [], []
        for name in spec.config.gripper_names:
            joint_obj = next((joint for joint in urdf_list if joint["name"] == name), None)
            lower.append(joint_obj["limit"]["lower"])
            upper.append(joint_obj["limit"]["upper"])
        constant = abs(float(lower[0]))
        scale = float(upper[0]) - float(lower[0])

        # Create engine_states (no agnostic states defined in this case)
        from eagerx_interbotix.xseries.pybullet.enginestates import PbXseriesGripper, LinkColorState
        from eagerx_pybullet.enginestates import JointState

        joints = spec.config.joint_names
        spec.engine.states.gripper = PbXseriesGripper.make(spec.config.gripper_names, constant, scale)
        spec.engine.states.position = JointState.make(joints=joints, mode="position")
        spec.engine.states.velocity = JointState.make(joints=joints, mode="velocity")
        spec.engine.states.color = LinkColorState.make()

        # Fix gripper if we are not controlling it.
        if "gripper_control" not in spec.config.actuators:
            spec.engine.states.gripper.fixed = True

        # Create sensor engine nodes
        from eagerx_pybullet.enginenodes import LinkSensor, JointSensor
        from eagerx_interbotix.xseries.pybullet.enginenodes import JointController, MoveItController

        pos_sensor = JointSensor.make("pos_sensor", rate=spec.sensors.position.rate, process=2, joints=joints, mode="position")
        vel_sensor = JointSensor.make("vel_sensor", rate=spec.sensors.velocity.rate, process=2, joints=joints, mode="velocity")
        ft_sensor = JointSensor.make(
            "ft_sensor", rate=spec.sensors.force_torque.rate, process=2, joints=["wrist_rotate"], mode="force_torque"
        )
        ee_pos_sensor = LinkSensor.make(
            "ee_pos_sensor",
            rate=spec.sensors.ee_pos.rate,
            links=[spec.config.gripper_link],
            mode="position",
        )
        ee_orn_sensor = LinkSensor.make(
            "ee_orn_sensor",
            rate=spec.sensors.ee_orn.rate,
            links=[spec.config.gripper_link],
            mode="orientation",
        )
        gripper_sensor = JointSensor.make(
            "gripper_sensor",
            rate=spec.sensors.gripper_position.rate,
            joints=spec.config.gripper_names[:1],
            mode="position",
        )

        # Create actuator engine nodes
        # Rate=None, but we will connect it to an actuator (thus will use the rate set in the agnostic specification)
        pos_control = JointController.make(
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            joints=joints,
            mode="position_control",
            vel_target=len(joints) * [0.0],
            pos_gain=len(joints) * [0.5],
            vel_gain=len(joints) * [1.0],
            max_vel=[0.5 * vel for vel in spec.config.vel_limit],
            max_force=len(joints) * [1.0],
        )
        vel_control = JointController.make(
            "vel_control",
            rate=spec.actuators.vel_control.rate,
            joints=joints,
            mode="velocity_control",
            vel_gain=len(joints) * [1.0],
            max_force=len(joints) * [2.0],  # todo: limit to 1.0?
            delay_state=True,
        )
        gripper = JointController.make(
            "gripper_control",
            rate=spec.actuators.gripper_control.rate,
            joints=spec.config.gripper_names,
            mode="position_control",
            vel_target=[0.0, 0.0],
            pos_gain=[0.5, 0.5],
            vel_gain=[1.0, 1.0],
            max_force=[0.5, 0.5],
        )
        moveit_to = MoveItController.make(
            "moveit_to",
            rate=spec.actuators.moveit_to.rate,
            joints=joints,
            vel_target=len(joints) * [0.0],
            pos_gain=len(joints) * [0.5],
            vel_gain=len(joints) * [1.0],
            max_vel=[0.5 * vel for vel in spec.config.vel_limit],
            max_force=len(joints) * [1.0],
        )
        from eagerx_interbotix.xseries.processor import MirrorAction

        gripper.inputs.action.processor = MirrorAction.make(index=0, constant=constant, scale=scale)

        # Connect all engine nodes
        graph.add(
            [
                pos_sensor,
                vel_sensor,
                ft_sensor,
                ee_pos_sensor,
                ee_orn_sensor,
                gripper_sensor,
                pos_control,
                vel_control,
                gripper,
                moveit_to,
            ]
        )
        graph.connect(source=pos_sensor.outputs.obs, sensor="position")
        graph.connect(source=vel_sensor.outputs.obs, sensor="velocity")
        graph.connect(source=ft_sensor.outputs.obs, sensor="force_torque")
        graph.connect(source=ee_pos_sensor.outputs.obs, sensor="ee_pos")
        graph.connect(source=ee_orn_sensor.outputs.obs, sensor="ee_orn")
        graph.connect(source=gripper_sensor.outputs.obs, sensor="gripper_position")
        graph.connect(source=moveit_to.outputs.status, sensor="moveit_status")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="vel_control", target=vel_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)
        graph.connect(actuator="moveit_to", target=moveit_to.inputs.action)

    @staticmethod
    @register.engine(RealEngine)
    def reality_engine(spec: ObjectSpec, graph: EngineGraph):
        """Engine-specific implementation (reality) of the object."""
        # Import any object specific entities for this engine
        import eagerx_interbotix.xseries.real  # noqa # pylint: disable=unused-import

        # Determine gripper min/max
        from eagerx_interbotix.xseries.real.enginestates import DummyState, CopilotStateReset

        spec.engine.states.position = CopilotStateReset.make(spec.config.name, spec.config.robot_type)
        spec.engine.states.velocity = DummyState.make()
        spec.engine.states.gripper = DummyState.make()
        spec.engine.states.color = DummyState.make()

        # Create sensor engine nodes
        from eagerx_interbotix.xseries.real.enginenodes import (
            XseriesGripper,
            XseriesSensor,
            XseriesArm,
            DummySensor,
            XseriesMoveIt,
        )

        joints = spec.config.joint_names
        robot_type = spec.config.robot_type
        arm_name = spec.config.arm_name
        # arm_name = robot_type

        # todo: set space to limits (pos=joint_limits, vel=vel_limits, effort=[-1, 1]?)
        pos_sensor = XseriesSensor.make(
            "pos_sensor",
            rate=spec.sensors.position.rate,
            joints=joints,
            mode="position",
            arm_name=arm_name,
            robot_type=robot_type,
        )
        ee_pos_sensor = XseriesSensor.make(
            "ee_pos_sensor",
            rate=spec.sensors.ee_pos.rate,
            joints=joints,
            mode="ee_position",
            arm_name=arm_name,
            robot_type=robot_type,
        )
        ee_orn_sensor = XseriesSensor.make(
            "ee_orn_sensor",
            rate=spec.sensors.ee_orn.rate,
            joints=joints,
            mode="ee_orientation",
            arm_name=arm_name,
            robot_type=robot_type,
        )
        vel_sensor = XseriesSensor.make(
            "vel_sensor",
            rate=spec.sensors.velocity.rate,
            joints=joints,
            mode="velocity",
            arm_name=arm_name,
            robot_type=robot_type,
        )
        gripper_sensor = XseriesSensor.make(
            "gripper_sensor",
            rate=spec.sensors.gripper_position.rate,
            joints=joints,
            mode="gripper_position",
            arm_name=arm_name,
            robot_type=robot_type,
        )

        # Create actuator engine nodes
        # todo: set space to limits (pos=joint_limits, vel=vel_limits, effort=[-1, 1]?)
        pos_control = XseriesArm.make(
            "pos_control",
            rate=spec.actuators.pos_control.rate,
            joints=joints,
            mode="position",
            profile_type="velocity",
            profile_acceleration=13,
            profile_velocity=131,
            kp_pos=800,
            kd_pos=1000,
            arm_name=arm_name,
            robot_type=robot_type,
        )
        vel_control = XseriesArm.make(
            "vel_control",
            rate=spec.actuators.vel_control.rate,
            joints=joints,
            mode="velocity",
            profile_type="time",
            profile_velocity=0,  # Must be 0, else mismatch!
            profile_acceleration=0,
            kp_pos=640,
            kd_pos=800,
            # kp_vel=5000,
            kp_vel=[5000, 5600, 5300, 5000, 5000, 5000],
            ki_vel=[400, 120, 120, 400, 300, 300],
            # kp_vel=[5000, 5600, 5300, 5000, 5000, 5000],
            # ki_vel=[350, 150, 150, 400, 250, 250],
            arm_name=arm_name,
            robot_type=robot_type,
        )
        moveit_to = XseriesMoveIt.make(
            "moveit_to",
            rate=spec.actuators.moveit_to.rate,
            arm_name=arm_name,
            robot_type=robot_type,
            joints=joints,
            vel_limit=[0.5 * c for c in spec.config.vel_limit],
            kp_pos=800,
            kd_pos=1000,
        )
        gripper = XseriesGripper.make(
            "gripper_control",
            rate=spec.actuators.gripper_control.rate,
            arm_name=arm_name,
            robot_type=robot_type,
            pressure=spec.config.pressure,
            pressure_lower=150,
            pressure_upper=350,
        )
        ft_sensor = DummySensor.make("ft_sensor", rate=spec.sensors.force_torque.rate)

        # Connect all engine nodes
        graph.add(
            [
                pos_sensor,
                vel_sensor,
                ee_pos_sensor,
                ee_orn_sensor,
                gripper_sensor,
                pos_control,
                vel_control,
                moveit_to,
                gripper,
                ft_sensor,
            ]
        )
        graph.connect(source=ft_sensor.outputs.obs, sensor="force_torque")
        graph.connect(source=pos_sensor.outputs.obs, sensor="position")
        graph.connect(source=vel_sensor.outputs.obs, sensor="velocity")
        graph.connect(source=ee_pos_sensor.outputs.obs, sensor="ee_pos")
        graph.connect(source=ee_orn_sensor.outputs.obs, sensor="ee_orn")
        graph.connect(source=gripper_sensor.outputs.obs, sensor="gripper_position")
        graph.connect(source=moveit_to.outputs.status, sensor="moveit_status")
        graph.connect(actuator="pos_control", target=pos_control.inputs.action)
        graph.connect(actuator="vel_control", target=vel_control.inputs.action)
        graph.connect(actuator="gripper_control", target=gripper.inputs.action)
        graph.connect(actuator="moveit_to", target=moveit_to.inputs.action)
