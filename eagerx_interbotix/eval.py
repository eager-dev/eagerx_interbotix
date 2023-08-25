# eagerx imports
import eagerx
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

    # Get parameters
    repetition = cfg["eval"]["repetition"]
    device = cfg["eval"]["device"]
    sim = cfg["eval"]["sim"]
    episodes = cfg["eval"]["episodes"]
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

    if sim:
        mode = "sim"
        engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=False, sync=True, real_time_factor=0)

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
        eval_file = eval_log_dir / "eval.yaml"

        # Check if evaluation already done
        if os.path.exists(eval_file):
            eval_results = yaml.safe_load(open(str(eval_file), "r"))
            if eval_results is not None and f"{mode}" in eval_results.keys():
                print(f"{mode} evaluation already done for {setting} {repetition}")
                continue
        else:
            # Create evaluation directory
            eval_log_dir.mkdir(parents=True, exist_ok=True)
            # Create evaluation file
            eval_file.touch()

        if cluster:
            graph_file = root / "exps" / "train" / "cluster_graphs" / f"graph_{setting}.yaml"
        else:
            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
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

        # Check if log dir exists
        if os.path.exists(LOAD_DIR):
            eval_env = create_env(cfg, repetition, graph, engine, backend)
            print("Loading model from: ", LOAD_DIR)
            model = sb.SAC.load(LOAD_DIR, env=eval_env, device=device)
        else:
            print(f"Model not found at {LOAD_DIR}.")
            continue

        print(f"Starting evaluation for {setting} {repetition}")
        eval_results = []
        obs_dict = {}
        action_dict = {}
        for i in tqdm(range(episodes)):
            obs_dict[i] = []
            action_dict[i] = []
            obs = eval_env.reset()
            obs_dict[i].append(obs)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_dict[i].append(action)
                obs, reward, done, info = eval_env.step(action)
                obs_dict[i].append(obs)
            # Calculate distance between box and goal
            box_pos = obs["achieved_goal"][:2]
            goal_pos = obs["desired_goal"][:2]
            dist = np.linalg.norm(box_pos - goal_pos)
            eval_results.append(dist)
        eval_results = np.array(eval_results)
        mean = np.mean(eval_results)
        std = np.std(eval_results)
        print(f"Mean: {mean}, Std: {std}")
        # Save results
        eval_dict = yaml.safe_load(open(str(eval_file), "r"))
        if eval_dict is None:
            eval_dict = {}
        eval_dict[mode] = {"mean": float(mean), "std": float(std), "results": eval_results.tolist()}
        with open(str(eval_file), "w") as f:
            yaml.dump(eval_dict, f)
        # Save observations and actions
        with open(str(eval_log_dir / f"{mode}_obs.pkl"), "wb") as f:
            pickle.dump(obs_dict, f)
        with open(str(eval_log_dir / f"{mode}_action.pkl"), "wb") as f:
            pickle.dump(action_dict, f)
        eval_env.shutdown()
