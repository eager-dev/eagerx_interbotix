# eagerx imports
import eagerx
import eagerx_interbotix
from eagerx_interbotix.env import ArmEnv
from eagerx_interbotix.goal_env import GoalArmEnv
from eagerx_pybullet.engine import PybulletEngine
from eagerx_reality.engine import RealEngine

# Common imports
import numpy as np
import os
import yaml
from pathlib import Path
from typing import Dict
import pickle
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

# Stable baselines imports
import stable_baselines3 as sb
from stable_baselines3.common.utils import set_random_seed
import gym.wrappers as w


def create_env(
    cfg: Dict, repetition: int, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec, backend: eagerx.specs.BackendSpec
):
    excl_z = cfg["eval"]["excl_z"]
    add_bias = cfg["eval"]["add_bias"]
    t_max = cfg["eval"]["t_max"]
    rate = cfg["eval"]["rate"]
    delay_min = cfg["settings"][setting]["delay_min"]
    delay_max = cfg["settings"][setting]["delay_max"]
    seed = 10**5 - repetition * 15
    set_random_seed(seed)

    env = ArmEnv(
        name=f"ArmEnv{seed}",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        exclude_z=excl_z,
        max_steps=int(t_max * rate),
        delay_min=float(delay_min),
        delay_max=float(delay_max),
        seed=seed,
    )
    goal_env = GoalArmEnv(env, add_bias=add_bias)
    eval_env = w.rescale_action.RescaleAction(goal_env, min_action=-1.0, max_action=1.0)
    return eval_env


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "eval.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    # Get parameters
    repetition = cfg["eval"]["repetition"]
    device = cfg["eval"]["device"]
    sim = cfg["eval"]["sim"]
    cluster = cfg["eval"]["cluster"]
    safe_rate = cfg["eval"]["safe_rate"]
    total_timesteps = cfg["eval"]["total_timesteps"]
    t_max = cfg["eval"]["t_max"]
    rate = cfg["eval"]["rate"]
    gui = cfg["eval"]["gui"]
    box_position_low = cfg["eval"]["box_position_low"]
    box_position_high = cfg["eval"]["box_position_high"]
    box_orientation_low = cfg["eval"]["box_orientation_low"]
    box_orientation_high = cfg["eval"]["box_orientation_high"]
    goal_position_low = cfg["eval"]["goal_position_low"]
    goal_position_high = cfg["eval"]["goal_position_high"]
    goal_orientation_low = cfg["eval"]["goal_orientation_low"]
    goal_orientation_high = cfg["eval"]["goal_orientation_high"]
    cam_index_rv = cfg["eval"]["cam_index_rv"]
    cam_index_ov = cfg["eval"]["cam_index_ov"]

    # Record parameters
    episodes = cfg["record"]["episodes"]
    video_width = cfg["record"]["video_width"]
    video_height = cfg["record"]["video_height"]
    video_width_ov = cfg["record"]["video_width_ov"]
    video_height_ov = cfg["record"]["video_height_ov"]
    overwrite = cfg["record"]["overwrite"]

    cam_translation_ov = [0.75, -0.049, 0.722]  # todo: set correct cam overview location
    cam_rotation_ov = [0.707, 0.669, -0.129, -0.192]  # todo: set correct cam overview location

    if sim:
        mode = "sim"
        engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=False, sync=True, real_time_factor=0)

        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()
    else:
        mode = "real"
        engine = RealEngine.make(rate=safe_rate, sync=True)

        from eagerx.backends.ros1 import ROS1
        backend = ROS1.make()

    for setting in cfg["settings"].keys():
        seed = repetition
        train_log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"
        LOAD_DIR = str(train_log_dir) + f"/rl_model_{total_timesteps}_steps.zip"
        eval_log_dir = root / "exps" / "eval" / "runs" / f"{setting}_{repetition}"
        record_file = eval_log_dir / f"{mode}_recording.mp4"

        # Check if recording already exists
        if os.path.exists(record_file) and not overwrite:
            print(f"Recording already exists at for {mode}, {setting}, {repetition}.")
            continue

        if cluster:
            graph_file = root / "exps" / "train" / "cluster_graphs" / f"graph_{setting}.yaml"
        else:
            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
        graph = eagerx.Graph.load(str(graph_file))

        solid = graph.get_spec("solid")
        solid.states.position.space.update(low=box_position_low, high=box_position_high)
        solid.states.orientation.space.update(low=box_orientation_low, high=box_orientation_high)

        goal = graph.get_spec("goal")
        goal.states.orientation.space.update(low=goal_orientation_low, high=goal_orientation_high)
        goal.states.position.space.update(low=goal_position_low, high=goal_position_high)

        solid.config.sensors = ["position", "yaw", "robot_view"]
        solid.config.cam_index = cam_index_rv

        # Create camera
        # from eagerx_interbotix.camera.objects import Camera
        #
        # cam = Camera.make(
        #     "cam",
        #     rate=rate,
        #     sensors=["image"],
        #     urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        #     optical_link="camera_color_optical_frame",
        #     calibration_link="camera_color_optical_frame",
        #     camera_index=cam_index_ov,
        #     fov=45.0,
        #     # render_shape=[video_height_ov, video_width_ov],
        # )
        # graph.add(cam)
        # cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        # cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)

        # Create overlay
        from eagerx_interbotix.overlay.node import Overlay

        # cam_extrinsics = {}
        # cam_extrinsics["camera_to_robot"] = {"translation": solid.config.cam_translation, "rotation": solid.config.cam_rotation}
        # overlay = Overlay.make("overlay", cam_intrinsics=solid.config.cam_intrinsics, cam_extrinsics=cam_extrinsics,
        #                        rate=rate, resolution=[video_height, video_width], caption="overview", ratio=0.3)
        # graph.add(overlay)

        # Connect
        graph.add_component(goal.sensors.orientation)
        # graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        # graph.connect(source=goal.sensors.orientation, target=overlay.inputs.goal_ori)
        # graph.connect(source=goal.sensors.position, target=overlay.inputs.goal_pos)
        # graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        # graph.render(source=overlay.outputs.image, rate=rate, encoding="bgr")
        graph.render(source=solid.sensors.robot_view, rate=rate, encoding="bgr")

        # Check if log dir exists
        if os.path.exists(LOAD_DIR):
            eval_env = create_env(cfg, repetition, graph, engine, backend)
            print("Loading model from: ", LOAD_DIR)
            model = sb.SAC.load(LOAD_DIR, env=eval_env, device=device)
        else:
            print(f"Model not found at {LOAD_DIR}.")
            continue

        print(f"Starting evaluation for {setting} {repetition}")
        video_buffer = []
        for i in tqdm(range(episodes)):
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                video_buffer.append(eval_env.render(mode="rgb_array"))
        clip = ImageSequenceClip(video_buffer, fps=rate)
        clip.write_videofile(str(record_file), fps=rate)
        eval_env.shutdown()
