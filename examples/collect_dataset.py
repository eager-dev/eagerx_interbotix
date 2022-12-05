import eagerx
import eagerx_interbotix

# Other
import yaml
import gym.wrappers as w
import stable_baselines3 as sb
import os
import h5py
import pathlib
import sys

sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")


NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
ROOT_DIR = pathlib.Path(eagerx_interbotix.__file__).parent.parent.resolve()
LOG_DIR = ROOT_DIR / "logs" / f"{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    CAM_PATH = ROOT_DIR / "assets" / "calibrations"
    CAM_INTRINSICS = "logitech_c920.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)

    sync = True
    must_render = True
    T_max = 15.0  # [sec]
    rate = 10
    render_rate = rate
    safe_rate = 10
    dataset_size = 100
    image_width, image_height = 64, 64

    # Camera settings
    cam_index_ov = 1
    cam_translation_ov = [0.8, 0, 0.8]  # todo: set correct cam overview location
    cam_rotation_ov = [-0.6830127, -0.6830127, 0.1830127, 0.1830127]  # todo: set correct cam overview location
    cam_render_shape = [image_height, image_width]

    # Create dataset file
    f = h5py.File(LOG_DIR / f"dataset_{dataset_size}.hdf5", "w")
    image_dataset = f.create_dataset("img", (dataset_size, image_height, image_width, 3), dtype="uint")
    boxpos_dataset = f.create_dataset("box_pos", (dataset_size, 3), dtype="float")
    boxyaw_dataset = f.create_dataset("box_yaw", (dataset_size, 1), dtype="float")
    goalpos_dataset = f.create_dataset("goal_pos", (dataset_size, 3), dtype="float")
    goalyaw_dataset = f.create_dataset("goal_yaw", (dataset_size, 1), dtype="float")

    # Load graph
    graph = eagerx.Graph.load(f"{LOG_DIR}/{GRAPH_FILE}")

    # Use correct workspace in safety filter
    safe = graph.get_spec("safety")
    safe.config.collision.workspace = "eagerx_interbotix.safety.workspaces/exclude_ground"

    # Modify aruco rate
    solid = graph.get_spec("solid")
    solid.sensors.robot_view.rate = 10
    solid.sensors.position.rate = 10
    solid.sensors.orientation.rate = 10
    solid.config.cam_intrinsics = cam_intrinsics

    # Modify goal
    goal = graph.get_spec("goal")
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    goal.states.position.space.update(low=[x, -y - dy, 0], high=[x + dx, y + dy, 0])

    # Add rendering
    if must_render:
        # Create camera
        from eagerx_interbotix.camera.objects import Camera

        cam = Camera.make(
            "cam",
            rate=render_rate,
            sensors=["image"],
            urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
            optical_link="camera_color_optical_frame",
            calibration_link="camera_color_optical_frame",
            camera_index=cam_index_ov,
            render_shape=cam_render_shape,
            fov=45.0,
        )
        graph.add(cam)
        cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)

        # Connect
        graph.render(source=cam.sensors.image, rate=render_rate, encoding="bgr")

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Make backend
    from eagerx.backends.single_process import SingleProcess

    SingleProcess.MIN_THREADS = 100
    backend = SingleProcess.make()

    # Define environment
    from eagerx_interbotix.env import ArmEnv
    from eagerx_interbotix.goal_env import GoalArmEnv

    # Initialize env
    env = ArmEnv(
        name="ArmEnv",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        add_bias=False,
        exclude_z=False,
        max_steps=int(T_max * rate),
    )
    sb_env = GoalArmEnv(env, add_bias=False)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, verbose=1)

    # Evaluate
    data_i = 0
    eps = 0
    while True:
        step = 0
        eps += 1
        print(f"Episode: {eps}, data index: {data_i}")
        obs, done, frames = sb_env.reset(), False, []
        while not done and data_i < dataset_size:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
            rgb = env.render("rgb_array")
            if len(rgb.shape) > 0 and rgb.shape[0] > 0 and step % 2 == 0:
                image_dataset[data_i] = rgb
                boxpos_dataset[data_i] = obs["achieved_goal"][:-1]
                boxyaw_dataset[data_i] = obs["achieved_goal"][-1]
                goalpos_dataset[data_i] = obs["desired_goal"][:-1]
                goalyaw_dataset[data_i] = obs["desired_goal"][-1]
                data_i += 1
            step += 1
        if not data_i < dataset_size:
            break

    f.close()
