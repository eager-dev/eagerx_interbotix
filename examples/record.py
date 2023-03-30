import eagerx
import eagerx_interbotix

# Other
import yaml
import gym.wrappers as w
import stable_baselines3 as sb
import os
import pathlib
from moviepy.editor import ImageSequenceClip

# NAME = "IK_10hz_line_vel_2022-08-16-1657"
NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
# NAME = "IK_10hz_circle_2022-08-17-1559"
# STEPS = 1_100_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
ROOT_DIR = pathlib.Path(eagerx_interbotix.__file__).parent.parent.resolve()
LOG_DIR = ROOT_DIR / "logs" / f"{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    CAM_PATH = ROOT_DIR / "assets" / "calibrations"
    CAM_INTRINSICS = "logitech_c920.yaml"
    # CAM_INTRINSICS = "logitech_c170.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)

    # Camera settings
    cam_index_ov = 1
    # cam_translation_ov = [1.0, 0, 1.0]  # todo: set correct cam overview location
    # cam_rotation_ov = [-0.6830127, -0.6830127, 0.1830127, 0.1830127]  # todo: set correct cam overview location
    # cam_rotation_ov = [-0.6963643, -0.6963642, 0.1227878, 0.1227878]
    cam_translation_ov = [1.25*0.811, 1.5*0.527, 1.5*0.43]
    cam_rotation_ov = [0.321, 0.801, -0.466, -0.197]
    # cam_translation_ov = [0.8, 0, 0.8]  # todo: set correct cam overview location
    # cam_rotation_ov = [-0.6830127, -0.6830127, 0.1830127, 0.1830127]  # todo: set correct cam overview location


    sync = True
    must_render = True
    T_max = 15.0  # [sec]
    rate = 10
    render_rate = rate
    safe_rate = 10
    light_direction_low = [-50, -50, 0]
    light_direction_high = [50, 50, 50]
    robot_color_low = [0, 0, 0, 1]
    robot_color_high = [0.2, 0.2, 0.2, 1]
    box_color_low = [0.9*1, 0.9*0.388, 0.9*0.278, 1]
    box_color_high = [1, 1.1*0.388, 1.1*0.278, 1]
    goal_color_low = [0.9*0.278, 0.9*1, 0.9*0.388, 1]
    goal_color_high = [1.1*0.278, 1, 1.1*0.388, 1]

    # Load graph
    graph = eagerx.Graph.load(f"{LOG_DIR}/{GRAPH_FILE}")

    # Use correct workspace in safety filter
    safe = graph.get_spec("safety")
    safe.config.collision.workspace = "eagerx_interbotix.safety.workspaces/exclude_ground"

    # Set color robot
    vx300s = graph.get_spec("vx300s")
    vx300s.states.color.space.update(low=robot_color_low, high=robot_color_high)

    # Modify box state
    solid = graph.get_spec("solid")
    solid.states.color.space.update(low=box_color_low, high=box_color_high)

    # Modify goal
    goal = graph.get_spec("goal")
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
    goal.states.position.space.update(low=[x, -y - dy, 0], high=[x + dx, y + dy, 0])
    goal.states.color.space.update(low=goal_color_low, high=goal_color_high)

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
            fov=45.0,
            render_shape=[1080, 1080],
            light_direction_low=light_direction_low,
            light_direction_high=light_direction_high,
        )
        graph.add(cam)
        cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)

        # Connect
        graph.render(source=cam.sensors.image, rate=render_rate, encoding="bgr")

    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Add surface
    surface_urdf = ROOT_DIR / "eagerx_interbotix" / "solid" / "assets" / "surface.urdf"
    engine.add_object("surface", urdf=str(surface_urdf), baseOrientation=[0, 0, 0, 1])

    # backend = Ros1.make()
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
    # env.render()
    sb_env = GoalArmEnv(env, add_bias=False)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, device="cuda", verbose=1)

    # Evaluate
    eps = 0
    video_buffer = []
    for i in range(10):
        step = 0
        obs, done, frames = sb_env.reset(), False, []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
            video_buffer.append(env.render("rgb_array"))
    clip = ImageSequenceClip(video_buffer, fps=25)
    clip.write_videofile(str(LOG_DIR / "recording_opendr.mp4"), fps=25)
    clip.close()

