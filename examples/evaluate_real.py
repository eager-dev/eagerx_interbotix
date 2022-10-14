import eagerx
import eagerx_interbotix

# Other
import yaml
import gym.wrappers as w
import stable_baselines3 as sb
import os

# NAME = "IK_10hz_line_vel_2022-08-16-1657"
NAME = "HER_force_torque_2022-10-12-1336"
STEPS = 1_600_000
# NAME = "IK_10hz_circle_2022-08-17-1559"
# STEPS = 1_100_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    CAM_PATH = "/home/jelle/eagerx_dev/eagerx_interbotix/assets/calibrations"
    # CAM_INTRINSICS = "logitech_c922.yaml"
    CAM_INTRINSICS = "logitech_c170.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)

    # Camera settings
    cam_index_rv = 0
    cam_index_ov = 1
    # cam_translation_rv = [0.811, 0.527, 0.43]
    cam_translation_rv = [0.864, -0.46, 0.525]
    # cam_rotation_rv = [0.321, 0.801, -0.466, -0.197]
    cam_rotation_rv = [0.813, 0.363, -0.177, -0.42]
    # translation = [0.864 - 0.46   0.525] | rotation = [0.813  0.363 - 0.177 - 0.42]
    cam_translation_ov = [0.75, -0.049, 0.722]  # todo: set correct cam overview location
    cam_rotation_ov = [0.707, 0.669, -0.129, -0.192]  # todo: set correct cam overview location

    sync = True
    add_bias = False
    exclude_z = False
    must_render = True
    save_video = False
    T_max = 15.0  # [sec]
    rate = 10
    render_rate = rate
    safe_rate = 10

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
    solid.config.cam_translation = cam_translation_rv
    solid.config.cam_rotation = cam_rotation_rv
    solid.config.cam_intrinsics = cam_intrinsics

    # Add rendering
    if must_render:
        solid.config.sensors = ["position", "yaw", "robot_view"]
        solid.config.cam_index = cam_index_rv

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
        )
        graph.add(cam)
        cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)

        # Create overlay
        from eagerx_interbotix.overlay.node import Overlay

        overlay = Overlay.make("overlay", rate=render_rate, resolution=[480, 480], caption="overview", ratio=0.3)
        graph.add(overlay)

        # Connect
        graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        graph.render(source=overlay.outputs.image, rate=render_rate, encoding="bgr")

    # Define engines
    from eagerx_reality.engine import RealEngine

    engine = RealEngine.make(rate=rate, sync=sync, process=eagerx.NEW_PROCESS)
    # from eagerx_pybullet.engine import PybulletEngine
    # engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Make backend
    from eagerx.backends.ros1 import Ros1

    backend = Ros1.make()
    # from eagerx.backends.single_process import SingleProcess
    # backend = SingleProcess.make()

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
        add_bias=add_bias,
        exclude_z=exclude_z,
        max_steps=int(T_max * rate),
    )
    env.render()
    sb_env = GoalArmEnv(env)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, device="cuda", verbose=1)

    # Setup rendering
    if save_video and must_render:
        from datetime import datetime
        import cv2
        from concurrent.futures import ThreadPoolExecutor

        cam_locations = dict(overview=dict(), robot_view=dict())
        for key, trans, rot in zip(
            ["overview", "robot_view"], (cam_translation_ov, cam_translation_rv), (cam_rotation_ov, cam_rotation_rv)
        ):
            cam_locations[key]["translation"] = trans
            cam_locations[key]["rotation"] = rot

        # Prepare video directory
        e = "pybullet" if "world_fn" in engine.config else "real"
        VIDEO_NAME = f"{e}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"
        VIDEO_DIR = f"{LOG_DIR}/videos/{VIDEO_NAME}"
        os.makedirs(VIDEO_DIR)

        # Save relevant parameters
        with open(f"{VIDEO_DIR}/camera.yaml", "w") as outfile:
            yaml.dump(cam_locations, outfile, default_flow_style=False)

        # Prepare frame saving
        executor = ThreadPoolExecutor(max_workers=1)

        # Prepare saving function
        def _save_video(eps_index, dir, frames, encoding="bgr"):
            if not len(frames) > 0:
                return

            FILE = f"{dir}/{eps_index}.mp4"
            try:
                # print(f"[START]: {FILE}")
                h, w = frames[0].shape[:2]
                out = cv2.VideoWriter(FILE, cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
                for f in frames:
                    f = f if not encoding == "bgr" else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    out.write(f)
                out.release()
            except Exception as e:
                print(f"[ERROR] {FILE} | {e}")
                return
            print(f"[SAVED]: {FILE}")

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done, frames = sb_env.reset(), False, []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
            if save_video and must_render:
                frames.append(env.render("rgb_array"))
        if save_video and must_render:
            executor.submit(_save_video, eps, VIDEO_DIR, frames)
