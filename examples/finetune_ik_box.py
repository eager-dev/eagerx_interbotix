import eagerx
import eagerx_interbotix
import os
import numpy as np
import stable_baselines3 as sb
import gym.wrappers as w
from datetime import datetime


NAME = "HER_force_torque"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"
LOAD_DIR = os.path.dirname(eagerx_interbotix.__file__) + "/../logs/HER_force_torque_2022-10-13-1836"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    n_procs = 1
    sync = True
    add_bias = False
    excl_z = False
    must_render = True
    reset = True
    save_video = False
    T_max = 10.0  # [sec]
    rate = 10
    safe_rate = 20
    MUST_LOG = True
    MUST_TEST = False
    gui = True
    max_delay = 0.4

    graph = eagerx.Graph.load(f"{LOAD_DIR}/{GRAPH_FILE}")
    safe = graph.get_spec("safety")
    vx300s = graph.get_spec("vx300s")

    # Add delay
    graph.disconnect(source=safe.outputs.filtered, target=vx300s.actuators.vel_control)
    graph.connect(source=safe.outputs.filtered, target=vx300s.actuators.vel_control, delay=0.25)

    # Define environment
    from eagerx_interbotix.env import ArmEnv
    from eagerx_interbotix.goal_env import GoalArmEnv

    from eagerx.backends.ros1 import Ros1
    backend = Ros1.make()

    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=gui, egl=True, sync=True, real_time_factor=0)

    env = ArmEnv(
        name=f"ArmEnv",
        rate=rate,
        graph=graph,
        engine=engine,
        backend=backend,
        exclude_z=excl_z,
        max_steps=int(T_max * rate),
        delay=max_delay,
    )
    goal_env = GoalArmEnv(env, add_bias=add_bias)
    train_env = w.rescale_action.RescaleAction(goal_env, min_action=-1.0, max_action=1.0)

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
    obs_space = train_env.observation_space

    # Evaluate
    if MUST_TEST:
        for eps in range(5000):
            print(f"Episode {eps}")
            _, done = train_env.reset(), False
            done = np.array([done], dtype="bool") if isinstance(done, bool) else done
            while not done.all():
                action = train_env.action_space.sample()
                obs, reward, done, info = train_env.step(action)

    # Create experiment directory
    total_steps = 1_600_000
    model = sb.SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=sb.HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
            # we have to manually specify the max number of steps per episode
            max_episode_length=int(T_max * rate),
            online_sampling=True,
        ),
        device="cuda",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    model.set_parameters(f"{LOAD_DIR}/rl_model_1600000_steps")
    model.learn(total_steps, callback=checkpoint_callback)
