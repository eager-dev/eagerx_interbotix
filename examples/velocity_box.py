import eagerx
import eagerx_interbotix
from eagerx.wrappers.flatten import Flatten
import numpy as np
import gym.wrappers as w
import stable_baselines3 as sb
from datetime import datetime
import os

NAME = "safety_bias"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"

# todo: TODAY
# todo: Solid: Aruco EngineNode --> Use cam_to_base to convert box_pose_cam to box_pose_base.
# todo: Solid: Camera EngineNode --> Initialize at location of calibration
# todo: Pybullet: Improve camera placement
# todo: Pybullet: Stop using RealSense camera in Camera Object?
# todo: Calibration: Create camera calibration routine
# todo: Calibration: Get correct ee_to_base transformation.
# todo: Calibration: Perform calibration with gripper_link instead of arm_link.
# todo: EAGERx: learn policy in Cartesian space.
# todo: EAGERx: Remove z measurement and randomize height of box --> should make it choose sides instead of top.
# todo: Copilot: For what link does Interbotix calculate the ee_pose?
# todo: Copilot Add gripper control to copilot GUI.
# todo: Copilot: Velocity control, what happens if robot arm is blocked? --> Overload...
# todo: Copilot: Monitor effort and stop() if too high?
# todo: Copilot: If Hardware error, stop(), smart reboot.
# todo: Copilot: Check write_commands based on mode + vel_lim & joint limits?
# todo: Copilot: Do not overwrite ProfileAcceleration/ProfileVelocity when using moveit --> avoids toggling torque.
# todo: Copilot: Call change motor registers when modifying ProfileAcceleration/ProfileVelocity.

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Define rate
    n_procs = 4
    rate = 20
    safe_rate = 20
    max_steps = 300
    MUST_LOG = True
    MUST_TEST = False

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create solid object
    from eagerx_interbotix.solid.solid import Solid
    import yaml
    urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    cam_path = os.path.dirname(eagerx_interbotix.__file__) + "/../assets/calibrations"
    cam_name = "logitech_camera"
    cam_extr_name = "2022-08-10-1413_logitech_camera_vx300s_inaccurate_0_029error"
    with open(f"{cam_path}/{cam_name}.yaml", "r") as f:
        cam_intrinsics = yaml.safe_load(f)
    with open(f"{cam_path}/{cam_extr_name}.yaml", "r") as f:
        cam_extrinsics = yaml.safe_load(f)
    cam_translation = cam_extrinsics["camera_to_robot"]["translation"]
    cam_rotation = cam_extrinsics["camera_to_robot"]["rotation"]
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
    x, y, z, dx, dy = 0.4, 0.2, 0.05, 0.03, 0.03
    solid.sensors.position.space.update(low=[-1, -1, 0], high=[1, 1, z + 0.1])
    solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    solid.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    solid.states.position.space.update(low=[x - dx, -y - dy, z], high=[x + dx, -y + dy, z])
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
    goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    goal.states.position.space.update(low=[x, y, z], high=[x, y, z])
    graph.add(goal)

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    robot_type = "vx300s"
    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "velocity", "ee_pos"],
        actuators=["vel_control"],
        states=["position", "velocity", "gripper"],
        rate=rate,
    )
    arm.states.gripper.space.update(low=[0.], high=[0.])  # Set gripper to closed position
    graph.add(arm)

    # Create safety node
    from eagerx_interbotix.safety.node import SafeVelocityControl
    c = arm.config
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
    graph.add(safe)

    # Connecting observations
    graph.connect(source=arm.sensors.position, observation="joints")
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
    graph.connect(source=arm.sensors.velocity, observation="velocity")
    graph.connect(source=solid.sensors.position, observation="solid")
    graph.connect(source=solid.sensors.yaw, observation="yaw")
    graph.connect(source=goal.sensors.position, observation="goal")
    # Connecting actions
    graph.connect(action="velocity", target=safe.inputs.goal)
    # Connecting safety filter to arm
    graph.connect(source=arm.sensors.position, target=safe.inputs.position)
    graph.connect(source=arm.sensors.velocity, target=safe.inputs.velocity)
    graph.connect(source=safe.outputs.filtered, target=arm.actuators.vel_control)

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
    class ArmEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, max_steps: int):
            self.steps = 0
            self.max_steps = max_steps

            # Bias
            self.solid_bias = None
            self.yaw_bias = None

            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start)

        def step(self, action):
            # Step the environment
            self.steps += 1
            info = dict()
            obs = self._step(action)

            # Calculate reward
            yaw = obs["yaw"]
            ee_pos = obs["ee_position"][0]
            goal = obs["goal"][0]
            can = obs["solid"][0]
            vel = obs["velocity"][0]
            des_vel = action["velocity"]
            # Penalize distance of the end-effector to the object
            rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - can) - 0.05)
            # Penalize distance of the object to the goal
            rwd_dist = 3.0 * -np.linalg.norm(goal - can)
            # Penalize actions (indirectly, by punishing the angular velocity.
            rwd_ctrl = 0.1 * -np.linalg.norm(des_vel - vel)
            rwd = rwd_dist + rwd_ctrl + rwd_near
            # Print rwd build-up
            # msg = f"rwd={rwd: .2f} | near={100*rwd_near/rwd: .1f} | dist={100*rwd_dist/rwd: .1f} | ctrl={100*rwd_ctrl/rwd: .1f}"
            # print(msg)
            # Determine done flag
            if self.steps > self.max_steps:  # Max steps reached
                done = True
                info["TimeLimit.truncated"] = True
            else:
                done = False | (np.linalg.norm(can[:2]) > 1.0)  # Can is out of reach
                if done:
                    rwd = -50

            # Simulate bias in observations.
            obs["solid"] += self.solid_bias
            obs["yaw"] = (obs["yaw"] + self.yaw_bias) % (np.pi / 2)
            return obs, rwd, done, info

        def reset(self):
            # Reset steps counter
            self.steps = 0

            # Sample bias
            self.solid_bias = 0.02*(2*np.random.random((3,)) - 1)
            self.yaw_bias = (0.1*np.pi/2)*(2*np.random.random() - 1)

            # Sample states
            states = self.state_space.sample()

            # Sample new starting state (at least 17 cm from goal)
            radius = 0.17
            while True:
                solid_pos = self.state_space["solid/position"].sample()
                goal_pos = self.state_space["goal/position"].sample()
                if np.linalg.norm(solid_pos[:2] - goal_pos[:2]) > radius:
                    states["solid/position"] = solid_pos
                    break

            # Set initial position state
            if "solid/aruco/position" in states:
                self.state_space["solid/aruco/position"] = solid_pos

            # Perform reset
            obs = self._reset(states)

            # Simulate bias in observations.
            obs["solid"] += self.solid_bias
            obs["yaw"] = (obs["yaw"] + self.yaw_bias) % (np.pi / 2)
            return obs


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
            env = ArmEnv(name=f"ArmEnv_{rank}", rate=rate, graph=graph, engine=engine, backend=backend, force_start=True, max_steps=max_steps)
            env = Flatten(env)
            env = w.rescale_action.RescaleAction(env, min_action=-1.5, max_action=1.5)
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
