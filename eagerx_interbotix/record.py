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
    add_bias = cfg["eval"]["add_bias"] if cfg["eval"]["sim"] else False
    t_max = cfg["eval"]["t_max"]
    rate = cfg["eval"]["rate"]
    delay_min = cfg["settings"][setting]["delay_min"] if cfg["eval"]["sim"] else 0
    delay_max = cfg["settings"][setting]["delay_max"] if cfg["eval"]["sim"] else 0
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
        eval=True,
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

    CAM_PATH = Path(__file__).parent.parent / "assets" / "calibrations"
    CAM_INTRINSICS = "logitech_c170_2023_02_17.yaml"
    CAM_INTRINSICS_OV = Path(__file__).parent.parent / "assets" / "logitech_c920" / "camera.yaml"
    CAM_EXTRINSICS = "eye_hand_calibration_2023-05-31-1111.yaml"
    CAM_EXTRINSICS_OV = "eye_hand_calibration_2023-06-06-0959.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)
    with open(f"{CAM_PATH}/{CAM_EXTRINSICS}", "r") as f:
        cam_extrinsics = yaml.safe_load(f)
    with open(f"{CAM_PATH}/{CAM_EXTRINSICS_OV}", "r") as f:
        cam_extrinsics_ov = yaml.safe_load(f)
    with open(f"{CAM_INTRINSICS_OV}", "r") as f:
        cam_intrinsics_ov = yaml.safe_load(f)

    # Camera settings
    cam_translation_rv = cam_extrinsics["camera_to_robot"]["translation"]
    cam_rotation_rv = cam_extrinsics["camera_to_robot"]["rotation"]
    cam_translation_ov = cam_extrinsics_ov["camera_to_robot"]["translation"]
    cam_rotation_ov = cam_extrinsics_ov["camera_to_robot"]["rotation"]

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

    if sim:
        mode = "sim"
        engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=False, sync=True, real_time_factor=0)

        surfuce_urdf = root / "eagerx_interbotix" / "solid" / "assets" / "table" / "table.urdf"
        engine.add_object(
            "surface", urdf=str(surfuce_urdf), baseOrientation=[0, 0, 0.7071068, 0.7071068], basePosition=[0.3, -0.3, -0.649]
        )

        from eagerx.backends.single_process import SingleProcess

        backend = SingleProcess.make()
    else:
        mode = "real"
        engine = RealEngine.make(rate=safe_rate, sync=True)

        from eagerx.backends.ros1 import Ros1

        backend = Ros1.make()

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
        if not os.path.exists(graph_file):
            print(f"Graph {graph_file} does not exist.")
            continue
        graph = eagerx.Graph.load(str(graph_file))

        safe = graph.get_spec("safety")
        # Update safety node for real robot
        if not sim:
            safe.config.collision.workspace = "eagerx_interbotix.safety.workspaces/exclude_ground_minus_2cm"
        elif "safety_filter" in cfg["settings"][setting].keys() and not cfg["settings"][setting]["safety_filter"]:
            # Remove safety filter
            graph.remove(safe)

            ik = graph.get_spec("ik")
            arm = graph.get_spec("vx300s")
            graph.connect(source=ik.outputs.dtarget, target=arm.actuators.vel_control)

        solid = graph.get_spec("solid")
        solid.states.position.space.update(low=box_position_low, high=box_position_high)
        solid.states.orientation.space.update(low=box_orientation_low, high=box_orientation_high)
        solid.config.cam_translation = cam_translation_rv
        solid.config.cam_rotation = cam_rotation_rv

        solid.gui(engine=PybulletEngine)

        goal = graph.get_spec("goal")
        goal.states.orientation.space.update(low=goal_orientation_low, high=goal_orientation_high)
        goal.states.position.space.update(low=goal_position_low, high=goal_position_high)

        vx300s = graph.get_spec("vx300s")
        target_pos = vx300s.states.position.space.low

        # Add reset node
        if "ik" in cfg["settings"][setting].keys() and not cfg["settings"][setting]["ik"]:
            from eagerx_interbotix.reset.node import MoveUpVelControl

            reset_node = MoveUpVelControl.make("reset", rate=rate, target_pos=vx300s.states.position.space.low)
            graph.add(reset_node)
            graph.connect(action="joint_vel", target=reset_node.feedthroughs.joint_vel)
            graph.disconnect(action="joint_vel", target=safe.inputs.goal)
            graph.connect(source=reset_node.outputs.joint_vel, target=safe.inputs.goal)
            graph.connect(source=vx300s.sensors.position, target=reset_node.inputs.current_pos)
            graph.connect(source=vx300s.sensors.velocity, target=reset_node.inputs.current_vel)
            graph.connect(source=vx300s.states.velocity, target=reset_node.targets.velocity)
        else:
            from eagerx_interbotix.reset.node import MoveUp

            ik = graph.get_spec("ik")

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
        solid = graph.get_spec("solid")
        solid.states.position.space.update(low=box_position_low, high=box_position_high)
        solid.states.orientation.space.update(low=box_orientation_low, high=box_orientation_high)

        goal = graph.get_spec("goal")
        goal.states.orientation.space.update(low=goal_orientation_low, high=goal_orientation_high)
        goal.states.position.space.update(low=goal_position_low, high=goal_position_high)
        graph.add_component(goal.sensors.orientation)

        solid.config.sensors = ["position", "yaw", "robot_view"]
        solid.config.cam_index = cam_index_rv
        # if not sim:
        # Create camera
        from eagerx_interbotix.camera.objects import Camera

        light_direction = [0, 0, 3]
        cam = Camera.make(
            "cam",
            rate=rate,
            sensors=["image"],
            urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
            optical_link="camera_color_optical_frame",
            calibration_link="camera_color_optical_frame",
            camera_index=cam_index_ov,
            fov=35.0,
            light_direction_low=light_direction,
            light_direction_high=light_direction,
            render_shape=[video_width_ov, video_height_ov],
        )
        graph.add(cam)
        cam.states.pos.space.update(low=cam_translation_ov, high=cam_translation_ov)
        cam.states.orientation.space.update(low=cam_rotation_ov, high=cam_rotation_ov)

        # Create overlay
        from eagerx_interbotix.overlay.node import Overlay

        overlay = Overlay.make(
            "overlay",
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics_overview=cam_intrinsics_ov,
            cam_extrinsics_overview=cam_extrinsics_ov,
            rate=rate,
            resolution=[video_height, video_width],
            caption="overview",
            ratio=0.3,
        )
        graph.add(overlay)

        # Connect
        graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        graph.connect(source=goal.sensors.orientation, target=overlay.inputs.goal_ori)
        graph.connect(source=goal.sensors.position, target=overlay.inputs.goal_pos)
        graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        graph.render(source=overlay.outputs.image, rate=rate, encoding="bgr")

        graph.gui()
        # else:
        #     graph.render(source=solid.sensors.robot_view, rate=rate, encoding="bgr")

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
        # from eagerx_interbotix.overlay.node import Overlay

        # cam_extrinsics = {}
        # cam_extrinsics["camera_to_robot"] = {"translation": solid.config.cam_translation, "rotation": solid.config.cam_rotation}
        # overlay = Overlay.make("overlay", cam_intrinsics=solid.config.cam_intrinsics, cam_extrinsics=cam_extrinsics,
        #                        rate=rate, resolution=[video_height, video_width], caption="overview", ratio=0.3)
        # graph.add(overlay)

        # Connect
        # graph.add_component(goal.sensors.orientation)
        # graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        # graph.connect(source=goal.sensors.orientation, target=overlay.inputs.goal_ori)
        # graph.connect(source=goal.sensors.position, target=overlay.inputs.goal_pos)
        # graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        # graph.render(source=overlay.outputs.image, rate=rate, encoding="bgr")
        # graph.render(source=solid.sensors.robot_view, rate=rate, encoding="bgr")

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
                rgb = eval_env.render(mode="rgb_array")
                if np.sum(rgb) > 0:
                    video_buffer.append(eval_env.render(mode="rgb_array"))
        clip = ImageSequenceClip(video_buffer, fps=rate)
        clip.write_videofile(str(record_file), fps=rate)
        eval_env.shutdown()
