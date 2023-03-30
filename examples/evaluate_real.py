import eagerx
import eagerx_interbotix

# Other
import yaml
import gym.wrappers as w
import stable_baselines3 as sb
import os
from pathlib import Path


# NAME = "HER_force_torque_2022-10-13-1836"
# NAME = "2023-02-21-2120_0.1_0.2"
# NAME = "2023-02-28-1234_0.1_0.2_no_ori"
NAME = "baseline"
REPETITION = 0
# STEPS = 1_600_000
# STEPS = 1_025_000
STEPS = 2_250_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
# LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../exps/train/runs/{NAME}_{REPETITION}"
GRAPH_FILE = os.path.dirname(eagerx_interbotix.__file__) + f"/../exps/train/graphs/graph_{NAME}.yaml"
# GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    CAM_PATH = Path(__file__).parent.parent / "assets" / "calibrations"
    # CAM_INTRINSICS = "logitech_c920.yaml"
    # CAM_INTRINSICS = "logitech_c170.yaml"
    CAM_INTRINSICS = "logitech_c170_2023_02_17.yaml"
    CAM_EXTRINSICS = "eye_hand_calibration_2023-02-22-1128.yaml"
    # CAM_EXTRINSICS = "eye_hand_calibration_2023-03-20-1037.yaml"
    # CAM_EXTRINSICS = "eye_hand_calibration_2023-03-01-0929.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)
    with open(f"{CAM_PATH}/{CAM_EXTRINSICS}", "r") as f:
        cam_extrinsics = yaml.safe_load(f)

    # Camera settings
    cam_index_rv = 2
    cam_index_ov = 4
    cam_translation_rv = cam_extrinsics["camera_to_robot"]["translation"]
    cam_rotation_rv = cam_extrinsics["camera_to_robot"]["rotation"]
    cam_translation_ov = [0.75, -0.049, 0.722]  # todo: set correct cam overview location
    cam_rotation_ov = [0.707, 0.669, -0.129, -0.192]  # todo: set correct cam overview location

    sync = False
    add_bias = False
    exclude_z = False
    must_render = True
    reset = True
    save_video = False
    T_max = 15.0  # [sec]
    rate = 10
    render_rate = rate
    safe_rate = 10
    sim = False
    # Load graph
    graph = eagerx.Graph.load(f"{GRAPH_FILE}")

    # Use correct workspace in safety filter
    safe = graph.get_spec("safety")
    safe.config.collision.workspace = "eagerx_interbotix.safety.workspaces/exclude_ground_minus_25mm"
    # safe.config.vel_limit = [x * 0.95 for x in safe.config.vel_limit]
    safe.config.margin = 0.01

    # Modify aruco rate
    solid = graph.get_spec("solid")
    solid.sensors.robot_view.rate = 10
    solid.sensors.position.rate = 10
    solid.sensors.orientation.rate = 10
    solid.config.cam_translation = cam_translation_rv
    solid.config.cam_rotation = cam_rotation_rv
    solid.config.cam_intrinsics = cam_intrinsics

    goal = graph.get_spec("goal")
    x, y, z = 0.35, 0.15, 0
    # solid.states.position.space.update(low=[x, -y, 0.05], high=[x, -y, 0.05])
    solid.states.orientation.space.update(low=[1, 0, 0, 0], high=[1, 0, 0, 0])
    goal.config.sensors = ["position", "orientation", "yaw"]
    goal.states.orientation.space.update(low=[1, 0, 0, 0], high=[1, 0, 0, 0])
    goal.states.position.space.update(low=[x, y, 0.0], high=[x, y, 0.0])

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

        overlay = Overlay.make("overlay", cam_intrinsics=cam_intrinsics, cam_extrinsics=cam_extrinsics, rate=render_rate, resolution=[480, 480], caption="overview", ratio=0.3)
        graph.add(overlay)

        # Connect
        graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        graph.connect(source=goal.sensors.orientation, target=overlay.inputs.goal_ori)
        graph.connect(source=goal.sensors.position, target=overlay.inputs.goal_pos)
        graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        graph.render(source=overlay.outputs.image, rate=render_rate, encoding="bgr")

    if reset:
        from eagerx_interbotix.reset.node import MoveUp

        ik = graph.get_spec("ik")
        vx300s = graph.get_spec("vx300s")
        target_pos = vx300s.states.position.space.low

        from eagerx_interbotix.xseries.mr_descriptions import vx300s as mr

        reset_node = MoveUp.make("reset", rate=rate, Slist=mr.Slist.tolist(), M=mr.M.tolist(), target_pos=target_pos)
        graph.add(reset_node)
        graph.connect(action="dxyz", target=reset_node.feedthroughs.dxyz)
        graph.connect(action="dyaw", target=reset_node.feedthroughs.dyaw)
        graph.disconnect(action="dxyz", target=ik.inputs.dxyz)
        graph.disconnect(action="dyaw", target=ik.inputs.dyaw)
        graph.connect(source=reset_node.outputs.dxyz, target=ik.inputs.dxyz)
        graph.connect(source=reset_node.outputs.dyaw, target=ik.inputs.dyaw)
        graph.connect(source=vx300s.states.velocity, target=reset_node.targets.velocity)
        graph.connect(source=vx300s.sensors.ee_pos, target=reset_node.inputs.ee_position)
        graph.connect(source=vx300s.sensors.ee_orn, target=reset_node.inputs.ee_orientation)
    # graph.gui()

    # Define engines
    if sim:
        from eagerx_pybullet.engine import PybulletEngine
        engine = PybulletEngine.make(rate=rate, sync=sync, gui=True, egl=True, process=eagerx.NEW_PROCESS)

        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()
    else:
        from eagerx_reality.engine import RealEngine
        engine = RealEngine.make(rate=rate, sync=sync, process=eagerx.NEW_PROCESS)

        from eagerx.backends.ros1 import Ros1
        backend = Ros1.make()

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
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, verbose=1)

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
    obs, done, frames = sb_env.reset(), False, []
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
