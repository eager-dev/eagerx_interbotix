# eagerx imports
import eagerx
from eagerx_interbotix.env import ArmEnv
from eagerx_interbotix.goal_env import GoalArmEnv

# Common imports
import glob
import os
import yaml
from pathlib import Path
from typing import Dict

# Stable baselines imports
import stable_baselines3 as sb
from stable_baselines3.common.utils import set_random_seed
import gym.wrappers as w



def create_env(
    cfg: Dict, repetition: int, time_step: int, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec, backend: eagerx.specs.BackendSpec
):
    excl_z = cfg["train"]["excl_z"]
    add_bias = cfg["train"]["add_bias"]
    t_max = cfg["train"]["t_max"]
    rate = cfg["train"]["rate"]
    delay_min = cfg["settings"][setting]["delay_min"]
    delay_max = cfg["settings"][setting]["delay_max"]
    save_freq = cfg["train"]["save_freq"]
    seed = repetition * 500

    seed += (time_step // save_freq) * 15
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
    train_env = w.rescale_action.RescaleAction(goal_env, min_action=-1.0, max_action=1.0)
    return train_env


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Get root path
    root = Path(__file__).parent.parent

    # Load config
    cfg_path = root / "cfg" / "train.yaml"
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)

    # Get parameters
    repetition = cfg["train"]["repetition"]
    device = cfg["train"]["device"]
    cluster = cfg["train"]["cluster"]
    safe_rate = cfg["train"]["safe_rate"]
    total_timesteps = cfg["train"]["total_timesteps"]
    save_freq = cfg["train"]["save_freq"]
    t_max = cfg["train"]["t_max"]
    rate = cfg["train"]["rate"]
    gui = cfg["train"]["gui"]
    keep_models = cfg["train"]["keep_models"]
    keep_buffers = cfg["train"]["keep_buffers"]

    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=False, sync=True, real_time_factor=0)

    for setting in cfg["settings"].keys():
        seed = repetition
        log_dir = root / "exps" / "train" / "runs" / f"{setting}_{repetition}"

        if cluster:
            graph_file = root / "exps" / "train" / "cluster_graphs" / f"graph_{setting}.yaml"
        else:
            graph_file = root / "exps" / "train" / "graphs" / f"graph_{setting}.yaml"
        graph = eagerx.Graph.load(str(graph_file))

        # Check if log dir exists
        if os.path.exists(log_dir) and len(glob.glob(str(log_dir) + "/rl_model_*.zip")) > 0:
            # Get last model
            checkpoints = glob.glob(str(log_dir) + "/rl_model_*.zip")
            # Get checkpoint step
            checkpoint_steps = [int(checkpoint.split("_")[-2]) for checkpoint in checkpoints]
            checkpoint_steps.sort()
            LOAD_DIR = str(log_dir) + f"/rl_model_{checkpoint_steps[-1]}_steps"
            step = int(LOAD_DIR.split("_")[-2])
            if step >= total_timesteps:
                print("Model already trained")
                continue
            print("Loading model from: ", LOAD_DIR)
            train_env = create_env(cfg, repetition, step, graph, engine, backend)
            model = sb.SAC.load(LOAD_DIR, env=train_env, tensorboard_log=str(log_dir), device=device)
            model.load_replay_buffer(LOAD_DIR + "_replay_buffer")
        else:
            print("No model found, starting from scratch")
            LOAD_DIR = None
            step = 0
            train_env = create_env(cfg, repetition, step, graph, engine, backend)
            model = sb.SAC(
                "MultiInputPolicy",
                train_env,
                replay_buffer_class=sb.HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                    # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
                    # we have to manually specify the max number of steps per episode
                    max_episode_length=int(t_max * rate),
                    online_sampling=True,
                ),
                device=device,
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=seed,
            )

        while step < total_timesteps:
            model.learn(
                total_timesteps=save_freq,
                tb_log_name="logs",
                reset_num_timesteps=False,
            )
            step += save_freq
            model.save(str(log_dir) + f"/rl_model_{step}_steps")
            model.save_replay_buffer(str(log_dir) + f"/rl_model_{step}_steps_replay_buffer")
            train_env.shutdown()
            train_env = create_env(cfg, repetition, step, graph, engine, backend)
            del model
            LOAD_DIR = str(log_dir) + f"/rl_model_{step}_steps"
            model = sb.SAC.load(LOAD_DIR, env=train_env, tensorboard_log=str(log_dir))
            model.load_replay_buffer(LOAD_DIR + "_replay_buffer")

            # Delete previous model if not keeping
            prev_step = step - save_freq
            if not keep_models and os.path.exists(str(log_dir) + f"/rl_model_{prev_step}_steps.zip"):
                os.remove(str(log_dir) + f"/rl_model_{prev_step}_steps.zip")

            # Delete previous replay buffer if not keeping
            if not keep_buffers and os.path.exists(str(log_dir) + f"/rl_model_{prev_step}_steps_replay_buffer.pkl"):
                os.remove(str(log_dir) + f"/rl_model_{prev_step}_steps_replay_buffer.pkl")

