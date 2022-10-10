import eagerx
import eagerx_interbotix
from eagerx.wrappers.flatten import Flatten
import numpy as np
import gym.wrappers as w
import stable_baselines3 as sb
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


NAME = "IK_10hz_circle_yaw_kn"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Define rate
    # todo: make distance larger
    # todo: reduce bias
    # todo: increase offset in rwd_near (to account for the gripper length)?
    # todo: Penalize box flipping?
    n_procs = 4
    rate = 10  # 20
    safe_rate = 20
    T_max = 10.0  # [sec]
    add_bias = True
    excl_z = False  # todo: z appears to be necessary. How to avoid pushing?
    USE_POS_CONTROL = False
    MUST_LOG = False
    MUST_TEST = False

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
        sensors=["position"],
        states=["position", "orientation"],
    )
    goal.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0.15])
    graph.add(goal)

    # Linear goal
    # x, y, z = 0.35, 0.2, 0.05
    # dx, dy = 0.03, 0.03
    # solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    # solid.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    # solid.states.position.space.update(low=[x - dx, -y - dy, z], high=[x + dx, -y + dy, z])
    # goal.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0.15])
    # goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    # goal.states.position.space.update(low=[x, y, z], high=[x, y, z])

    # Circular goal
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    solid.states.orientation.space.update(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    goal.states.position.space.update(low=[x, y, z], high=[x, y, z])

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    robot_type = "vx300s"
    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "velocity", "ee_pos", "ee_orn"],
        actuators=[],
        states=["position", "velocity", "gripper"],
        rate=rate,
    )
    arm.states.gripper.space.update(low=[0.], high=[0.])  # Set gripper to closed position
    arm.states.position.space.low[-2] = np.pi / 2
    arm.states.position.space.high[-2] = np.pi / 2
    graph.add(arm)

    # Create IK node
    from eagerx_interbotix.ik.node import EndEffectorDownward
    import eagerx_interbotix.xseries.mr_descriptions as mrd

    robot_des = getattr(mrd, robot_type)
    c = arm.config
    ik = EndEffectorDownward.make("ik",
                                  rate,
                                  c.joint_names,
                                  robot_des.Slist.tolist(),
                                  robot_des.M.tolist(),
                                  c.joint_upper,
                                  c.joint_lower,
                                  max_dxyz=[0.2, 0.2, 0.2],  # 10 cm / sec
                                  max_dyaw=2 * np.pi / 2,    # 1/5 round / second
                                  )
    graph.add(ik)

    if USE_POS_CONTROL:
        safe = position_control(graph, arm, dict(source=ik.outputs.target), safe_rate)
    else:
        safe = velocity_control(graph, arm, dict(source=ik.outputs.dtarget), safe_rate)

    # Connecting observations
    graph.connect(source=arm.sensors.position, observation="joints")
    graph.connect(source=arm.sensors.velocity, observation="velocity")
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
    graph.connect(source=solid.sensors.position, observation="solid")
    graph.connect(source=solid.sensors.yaw, observation="yaw")
    graph.connect(source=goal.sensors.position, observation="goal")
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

        # translation=[ 0.75  -0.049  0.722] | rotation=[ 0.707  0.669 -0.129 -0.192]
        # translation = [0.788 0.009 0.681] | rotation = [0.674  0.682 - 0.221 - 0.177]
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


    def make_env(rank: int, use_ros: bool = True):
        gui = True if rank == 0 else False
        if rank == 0 and use_ros:
            from eagerx.backends.ros1 import Ros1
            backend = Ros1.make()
        else:
            from eagerx.backends.single_process import SingleProcess
            backend = SingleProcess.make()

        # Define engines
        from eagerx_pybullet.engine import PybulletEngine
        engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=True, sync=True, real_time_factor=0.0)
        # from eagerx_reality.engine import RealEngine
        # engine = RealEngine.make(rate=safe_rate, sync=True)

        def _init():
            graph.reload()
            env = ArmEnv(name=f"ArmEnv_{rank}",
                         rate=rate,
                         graph=graph,
                         engine=engine,
                         backend=backend,
                         exclude_z=excl_z,
                         max_steps=int(T_max * rate),
                         add_bias=add_bias)
            env = Flatten(env)
            env = w.rescale_action.RescaleAction(env, min_action=-1.0, max_action=1.0)
            # env.render()
            return env

        return _init

    # Use multi-processing
    if n_procs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        train_env = SubprocVecEnv([make_env(i) for i in range(n_procs)], start_method='spawn')
    else:
        train_env = make_env(rank=0, use_ros=False)()

    # Initialize model
    if MUST_LOG:
        os.mkdir(LOG_DIR)
        graph.save(f"{LOG_DIR}/graph.yaml")
        from stable_baselines3.common.callbacks import CheckpointCallback
        checkpoint_callback = CheckpointCallback(save_freq=25_000, save_path=LOG_DIR, name_prefix="rl_model")
    else:
        LOG_DIR = None
        checkpoint_callback = None

    # First train in simulation
    # train_env.render("human")

    # Evaluate
    if MUST_TEST:
        for eps in range(5000):
            print(f"Episode {eps}")
            _, done = train_env.reset(), False
            while not done:
                action = train_env.action_space.sample()
                obs, reward, done, info = train_env.step(action)

    # Create experiment directory
    total_steps = 1_600_000
    model = sb.SAC("MlpPolicy", train_env, device="cuda", verbose=1, tensorboard_log=LOG_DIR)
    model.learn(total_steps, callback=checkpoint_callback)

    # # First train in simulation
    # env.render("human")
    #
    # # Evaluate
    # for eps in range(5000):
    #     print(f"Episode {eps}")
    #     _, done = env.reset(), False
    #     while not done:
    #         action = env.action_space.sample()
    #         obs, reward, done, info = env.step(action)
    #         rgb = env.render("rgb_array")

