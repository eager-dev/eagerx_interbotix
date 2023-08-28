import eagerx
from eagerx.wrappers.flatten import Flatten
import eagerx_interbotix

# Other
import numpy as np
import gym.wrappers as w
import stable_baselines3 as sb
import os

# NAME = "space_box_dynamicsRandomization_2022-07-22-1313"
# NAME = "IK_10hz_line_vel_2022-08-16-1657"
NAME = "IK_10hz_circle_yaw_kn_2022-08-25-1638"
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

    # todo: Overwrite goal, solid, gripper positions
    # solid = graph.get_spec("solid")
    # solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    # solid.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    # solid.states.pos.space.update(low=[0.4 - 0.03, -0.2 - 0.03, 0.035], high=[0.4 + 0.03, -0.2 + 0.03, 0.035])
    # goal = graph.get_spec("goal")
    # goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    # goal.states.pos.space.update(low=[0.4, 0.2, 0.035], high=[0.4, 0.2, 0.035])
    # arm = graph.get_spec("viper")
    # arm.states.gripper.space.update(low=[0.], high=[0.])  # Set gripper to closed position

    # Define engines
    # from eagerx_reality.engine import RealEngine
    # engine = RealEngine.make(rate=rate, sync=True, process=eagerx.NEW_PROCESS)
    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0)

    # Make backend
    # from eagerx.backends.ros1 import Ros1
    # backend = Ros1.make()
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Define environment
    from eagerx_interbotix.env import ArmEnv

    # Initialize env
    env = ArmEnv(name="ArmEnv",
                 rate=rate,
                 graph=graph,
                 engine=engine,
                 backend=backend,
                 add_bias=add_bias,
                 exclude_z=exclude_z,
                 max_steps=int(T_max * rate))
    sb_env = Flatten(env)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.0, max_action=1.0)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, device="cuda", verbose=1)

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done = sb_env.reset(), False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
