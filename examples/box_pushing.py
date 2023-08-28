import eagerx
import eagerx_interbotix
import numpy as np
from datetime import datetime
import os


def position_control(_graph, _arm, source_goal, safe_rate):
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
        [0.2 * vl for vl in c.vel_limit],
        checks=3,
        collision=collision,
    )
    _graph.add(safe)

    # Connecting safety filter to arm
    _graph.connect(**source_goal, target=safe.inputs.goal)
    _graph.connect(source=_arm.sensors.position, target=safe.inputs.current)
    _graph.connect(source=safe.outputs.filtered, target=_arm.actuators.pos_control)

    return safe


def velocity_control(_graph, _arm, source_goal, safe_rate):
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
        [0.2 * vl for vl in c.vel_limit],
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


NAME = "HER_force_torque"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    n_procs = 1
    rate = 10  # 20
    safe_rate = 20
    T_max = 10.0  # [sec]
    add_bias = True
    excl_z = False
    USE_POS_CONTROL = False

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create solid object
    from eagerx_interbotix.solid.solid import Solid
    import yaml

    urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    cam_path = os.path.dirname(eagerx_interbotix.__file__) + "/../assets/calibrations"
    cam_name = "logitech_c170"
    with open(f"{cam_path}/{cam_name}.yaml", "r") as f:
        cam_intrinsics = yaml.safe_load(f)
    cam_translation = [0.811, 0.527, 0.43]
    cam_rotation = [0.321, 0.801, -0.466, -0.197]

    solid = Solid.make(
        "solid",
        urdf=urdf_path + "box.urdf",
        rate=rate,
        cam_translation=cam_translation,
        cam_rotation=cam_rotation,
        cam_index=2,
        cam_intrinsics=cam_intrinsics,
        # sensors=["position", "yaw", "robot_view"],  # select robot_view to render.
        sensors=["position", "yaw"],  # select robot_view to render.
        states=["position", "velocity", "orientation", "angular_vel", "lateral_friction"],
    )

    solid.sensors.position.space.update(low=[-1, -1, 0], high=[1, 1, 0.13])
    graph.add(solid)

    # Create solid goal
    from eagerx_interbotix.solid.goal import Goal

    goal = Goal.make(
        "goal",
        urdf=urdf_path + "box_goal.urdf",
        rate=rate,
        sensors=["position", "yaw"],
        states=["position", "orientation"],
    )
    goal.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0.15])
    graph.add(goal)

    # Circular goal
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    solid.states.orientation.space.update(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    goal.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])

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

    if USE_POS_CONTROL:
        safe = position_control(graph, arm, dict(source=ik.outputs.target), safe_rate)
    else:
        safe = velocity_control(graph, arm, dict(source=ik.outputs.dtarget), safe_rate)

    # Connecting observations
    graph.connect(source=arm.sensors.position, observation="joints")
    graph.connect(source=arm.sensors.velocity, observation="velocity")
    graph.connect(source=arm.sensors.force_torque, observation="force_torque")
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
    graph.connect(source=solid.sensors.position, observation="pos")
    graph.connect(source=solid.sensors.yaw, observation="yaw")
    graph.connect(source=goal.sensors.position, observation="pos_desired")
    graph.connect(source=goal.sensors.yaw, observation="yaw_desired")
    # Connect IK
    graph.connect(source=arm.sensors.position, target=ik.inputs.current)
    graph.connect(source=arm.sensors.ee_pos, target=ik.inputs.xyz)
    graph.connect(source=arm.sensors.ee_orn, target=ik.inputs.orn)
    # Connecting actions
    graph.connect(action="dxyz", target=ik.inputs.dxyz)
    graph.connect(action="dyaw", target=ik.inputs.dyaw)

    # Add rendering
    if "robot_view" in solid.config.sensors:
        # Create camera
        from eagerx_interbotix.camera.objects import Camera

        cam = Camera.make(
            "cam",
            rate=rate,
            sensors=["image"],
            urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
            optical_link="camera_color_optical_frame",
            calibration_link="camera_bottom_screw_frame",
            camera_index=0,  # todo: set correct index
        )
        graph.add(cam)
        # Create overlay
        from eagerx_interbotix.overlay.node import Overlay

        overlay = Overlay.make("overlay", rate=20, resolution=[480, 480], caption="robot view")
        graph.add(overlay)
        # Connect
        graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        graph.render(source=overlay.outputs.image, rate=20, encoding="bgr")

    # Define environment
    from eagerx_interbotix.env import ArmEnv
    from eagerx_interbotix.goal_env import GoalArmEnv

    from eagerx.backends.ros1 import Ros1
    backend = Ros1.make()

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, egl=True, sync=True, real_time_factor=0)

    env = ArmEnv(
        name="BoxPushEnv",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        exclude_z=excl_z,
        max_steps=int(T_max * rate),
    )
    goal_env = GoalArmEnv(env, add_bias=add_bias)

    # First train in simulation
    # goal_env.render("human")
    obs_space = goal_env.observation_space

    for eps in range(5000):
        print(f"Episode {eps}")
        _, _, done = goal_env.reset(), False
        done = np.array([done], dtype="bool") if isinstance(done, bool) else done
        while not done.all():
            action = goal_env.action_space.sample()
            obs, reward, terminated, truncated, info = goal_env.step(action)
