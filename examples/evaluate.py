import eagerx
import eagerx_interbotix

# Other
import gym.wrappers as w
import stable_baselines3 as sb
import os


NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    add_bias = False
    exclude_z = False
    T_max = 10.0  # [sec]
    rate = 10
    safe_rate = 20

    # Load graph
    graph = eagerx.Graph.load(f"{LOG_DIR}/{GRAPH_FILE}")
    graph.gui()

    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0)

    # Make backend
    from eagerx.backends.single_process import SingleProcess

    backend = SingleProcess.make()

    # Define environment
    from eagerx_interbotix.env import ArmEnv
    from eagerx_interbotix.goal_env import GoalArmEnv

    # Initialize env
    env = ArmEnv(
        name="ArmEnv", rate=rate, graph=graph, engine=engine, backend=backend, exclude_z=exclude_z, max_steps=int(T_max * rate)
    )
    sb_env = GoalArmEnv(env, add_bias=add_bias)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, verbose=1)

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done = sb_env.reset(), False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
