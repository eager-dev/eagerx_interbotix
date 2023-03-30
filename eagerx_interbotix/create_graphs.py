import eagerx
from pathlib import Path
import yaml
import numpy as np


def position_control(_graph, _arm, source_goal, safe_rate, vel_limit):
    # Add position control actuator
    if "pos_control" not in _arm.config.actuators:
        _arm.config.actuators.append("pos_control")

    # Create safety node
    from eagerx_interbotix.safety.node import SafePositionControl

    c = _arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
        # workspace="eagerx_interbotix.safety.workspaces/exclude_ground_minus_2m",
        margin=0.01,  # [cm]
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = SafePositionControl.make(
        "safety",
        safe_rate,
        c.joint_names,
        c.joint_upper,
        c.joint_lower,
        [vel_limit for _ in c.vel_limit],
        checks=3,
        collision=collision,
    )
    _graph.add(safe)

    # Connecting safety filter to arm
    _graph.connect(**source_goal, target=safe.inputs.goal)
    _graph.connect(source=_arm.sensors.position, target=safe.inputs.current)
    _graph.connect(source=safe.outputs.filtered, target=_arm.actuators.pos_control)

    return safe


def velocity_control(_graph, _arm, source_goal, safe_rate, vel_limit):
    # Add velocity control actuator
    if "vel_control" not in _arm.config.actuators:
        _arm.config.actuators.append("vel_control")

    # Create safety node
    from eagerx_interbotix.safety.node import SafeVelocityControl

    c = _arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
        # workspace="eagerx_interbotix.safety.workspaces/exclude_ground_minus_2m",
        margin=0.01,  # [cm]
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = SafeVelocityControl.make(
        "safety",
        safe_rate,
        c.joint_names,
        c.joint_upper,
        c.joint_lower,
        [vel_limit for _ in c.vel_limit],
        checks=3,
        collision=collision,
    )
    _graph.add(safe)

    # Connecting goal
    _graph.connect(**source_goal, target=safe.inputs.goal)
    # Connecting safety filter to arm
    _graph.connect(source=_arm.sensors.position, target=safe.inputs.position)
    _graph.connect(source=_arm.sensors.velocity, target=safe.inputs.velocity)
    _graph.connect(source=safe.outputs.filtered, target=arm.actuators.vel_control)

    return safe


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path("/home/jelle/eagerx_dev/eagerx_interbotix")

    # Load config
    cfg_path = root / "cfg" / "train.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    rate = cfg["train"]["rate"]
    safe_rate = cfg["train"]["safe_rate"]
    pos_control = cfg["train"]["pos_control"]
    vel_limit = cfg["train"]["vel_limit"]
    box_position_low = cfg["train"]["box_position_low"]
    box_position_high = cfg["train"]["box_position_high"]
    box_orientation_low = cfg["train"]["box_orientation_low"]
    box_orientation_high = cfg["train"]["box_orientation_high"]
    goal_position_low = cfg["train"]["goal_position_low"]
    goal_position_high = cfg["train"]["goal_position_high"]
    goal_orientation_low = cfg["train"]["goal_orientation_low"]
    goal_orientation_high = cfg["train"]["goal_orientation_high"]

    settings = cfg["settings"]

    for setting in settings:
        friction_low = cfg["settings"][setting]["friction_low"]
        friction_high = cfg["settings"][setting]["friction_high"]

        graph = eagerx.Graph.create()

        # Create solid box
        from eagerx_interbotix.solid.solid import Solid

        urdf_path = root / "eagerx_interbotix" / "solid" / "assets"
        intrinsics_path = root / "assets" / "calibrations" / "logitech_c170_2023_02_17.yaml"
        extrinsics_path = root / "assets" / "calibrations" / "eye_hand_calibration_2023-02-22-1128.yaml"
        with open(str(intrinsics_path), "r") as f:
            cam_intrinsics = yaml.safe_load(f)
        with open(str(extrinsics_path), "r") as f:
            cam_extrinsics = yaml.safe_load(f)
        cam_translation_rv = cam_extrinsics["camera_to_robot"]["translation"]
        cam_rotation_rv = cam_extrinsics["camera_to_robot"]["rotation"]

        solid = Solid.make(
            "solid",
            urdf=str(urdf_path / "box.urdf"),
            rate=rate,
            cam_translation=cam_translation_rv,
            cam_rotation=cam_rotation_rv,
            cam_index=2,
            cam_intrinsics=cam_intrinsics,
            sensors=["position", "yaw"],  # select robot_view to render.
            states=["position", "velocity", "orientation", "angular_vel", "lateral_friction"],
        )

        solid.sensors.position.space.update(low=[-1, -1, 0], high=[1, 1, 0.15])
        solid.states.lateral_friction.space.update(low=friction_low, high=friction_high)
        solid.states.orientation.space.update(low=box_orientation_low, high=box_orientation_high)
        solid.states.position.space.update(low=box_position_low, high=box_position_high)
        graph.add(solid)

        # Create solid goal
        from eagerx_interbotix.solid.goal import Goal

        goal = Goal.make(
            "goal",
            urdf=str(urdf_path / "box_goal.urdf"),
            rate=rate,
            sensors=["position"],
            states=["position", "orientation"],
        )
        goal.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0.15])
        goal.states.orientation.space.update(low=goal_orientation_low, high=goal_orientation_high)
        goal.states.position.space.update(low=goal_position_low, high=goal_position_high)
        graph.add(goal)

        # Create arm
        from eagerx_interbotix.xseries.xseries import Xseries

        robot_type = "vx300s"
        arm = Xseries.make(
            name=robot_type,
            robot_type=robot_type,
            sensors=["position", "velocity", "force_torque", "ee_pos", "ee_orn"],
            actuators=[],
            states=["position", "velocity", "gripper"],
            rate=rate,
        )
        arm.states.gripper.space.update(low=[0.0], high=[0.0])  # Set gripper to closed position
        arm.states.position.space.low[-2] = np.pi / 2
        arm.states.position.space.high[-2] = np.pi / 2
        graph.add(arm)

        # Create IK node
        from eagerx_interbotix.ik.node import EndEffectorDownward
        import eagerx_interbotix.xseries.mr_descriptions as mrd

        robot_des = getattr(mrd, robot_type)
        c = arm.config
        ik = EndEffectorDownward.make(
            "ik",
            rate,
            c.joint_names,
            robot_des.Slist.tolist(),
            robot_des.M.tolist(),
            c.joint_upper,
            c.joint_lower,
            max_dxyz=[0.2, 0.2, 0.2],  # 10 cm / sec
            max_dyaw=2 * np.pi / 2,  # 1/5 round / second
        )
        graph.add(ik)

        if pos_control:
            safe = position_control(graph, arm, dict(source=ik.outputs.target), safe_rate, vel_limit)
        else:
            safe = velocity_control(graph, arm, dict(source=ik.outputs.dtarget), safe_rate, vel_limit)

        # Connecting observations
        graph.connect(source=arm.sensors.position, observation="joints")
        graph.connect(source=arm.sensors.velocity, observation="velocity")
        graph.connect(source=arm.sensors.force_torque, observation="force_torque")
        graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
        graph.connect(source=solid.sensors.position, observation="pos")
        graph.connect(source=solid.sensors.yaw, observation="yaw")
        graph.connect(source=goal.sensors.position, observation="pos_desired")
        # Connect IK
        graph.connect(source=arm.sensors.position, target=ik.inputs.current)
        graph.connect(source=arm.sensors.ee_pos, target=ik.inputs.xyz)
        graph.connect(source=arm.sensors.ee_orn, target=ik.inputs.orn)
        # Connecting actions
        graph.connect(action="dxyz", target=ik.inputs.dxyz)
        graph.connect(action="dyaw", target=ik.inputs.dyaw)

        graph_dir = root / "exps" / "train" / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / f"graph_{setting}.yaml"
        graph.save(str(graph_path))
