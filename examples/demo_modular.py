# EAGERx imports
from eagerx.wrappers.flatten import Flatten
from eagerx.core.graph import Graph
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletEngine # noqa # pylint: disable=unused-import
import eagerx_interbotix  # Registers objects # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers engine # noqa # pylint: disable=unused-import

# Registers PybulletEngine
from eagerx_quadruped.helper import add_quadruped
from eagerx_interbotix.utils import add_manipulator

# Other
import numpy as np
import gym.wrappers as w
import os

# NAME = "sac_dynamicsRandomization_2022-04-13-1240"
# STEPS = 700000
NAME = "better_velControl_highCtrlCost_normalized_2022-04-12-1610"
STEPS = 900000
MODEL_NAME = f"{NAME}/model_{STEPS}"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{MODEL_NAME}"
if __name__ == "__main__":
    eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.WARN)

    # Define rate
    use_safety = False
    real_reset = False
    rate = 20
    safe_rate = 20
    engine_rate = 200
    max_steps = 200

    # Initialize empty graph
    graph = Graph.create()

    # Add quadruped
    for i in range(2):
        quad = add_quadruped(graph, f"quad_{i}", ["base_pos"], [-2, -0.5 + i, 0.33], base_orientation=[0, 0, 0, 1])
        graph.connect(source=quad.sensors.base_pos, observation=f"quad_{i}_pos")

    # Create camera
    # cam = eagerx.Object.make(
    #     "Camera",
    #     "cam",
    #     rate=rate,
    #     sensors=["rgb"],
    #     urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
    #     optical_link="camera_color_optical_frame",
    #     calibration_link="camera_bottom_screw_frame",
    # )
    # graph.add(cam)

    # Create solid object
    urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    solid = eagerx.Object.make(
        "Solid", "solid", urdf=urdf_path + "can.urdf", rate=rate, sensors=["pos"], base_pos=[0, 0.30, 0.05], fixed_base=False,
        states=["pos", "vel", "orientation", "angular_vel", "lateral_friction"]
    )
    solid.sensors.pos.space_converter.low = [0, -1, 0]
    solid.sensors.pos.space_converter.high = [1, 1, 0.15]
    solid.states.lateral_friction.space_converter.low = 0.4
    solid.states.lateral_friction.space_converter.high = 0.1
    graph.add(solid)

    # Connecting observations
    # graph.connect(source=solid.sensors.pos, observation="solid")

    # Create arm
    # for i, model_type in enumerate(["px100", "px150", "rx200", "wx250", "vx300s"]):
    #     arm = add_manipulator(graph, model_type, model_type, base_pos=[0, -1 + i*0.5, 0.], base_orientation=[0, 0, 0, 1])
    graph.gui()

    # Define engines
    # engine = Engine.make("RealEngine", rate=rate, sync=True, process=process.NEW_PROCESS)
    engine = eagerx.Engine.make("PybulletEngine", rate=engine_rate, gui=False, egl=True, sync=True, real_time_factor=0)

    # Define environment
    import gym
    from typing import Tuple, Dict

    class ComposedEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, force_start=True):
            super(ComposedEnv, self).__init__(name, rate, graph, engine, force_start=force_start)
            self.steps = None

        @property
        def observation_space(self) -> gym.spaces.Dict:
            return self._observation_space

        @property
        def action_space(self) -> gym.spaces.Dict:
            return self._action_space

        def reset(self):
            # Reset number of steps
            self.steps = 0

            # Sample desired states
            states = self.state_space.sample()

            # Sample new starting state (at least 17 cm from goal)
            radius = 0.17
            z = 0.03
            while True:
                can_pos = np.concatenate(
                    [
                        np.random.uniform(low=0, high=1.1 * radius, size=1),
                        np.random.uniform(low=-1.2 * radius, high=1.2 * radius, size=1),
                        [z],
                    ]
                )
                if np.linalg.norm(can_pos) > radius:
                    break
            if "solid/pos" in states:
                states["solid/pos"] = np.array([0.4, -0.2, z])

            # Set grippers to closed position
            for key in states.keys():
                if "gripper" in key:
                    states[key][0] = 0

            # Perform reset
            obs = self._reset(states)
            return obs

        def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
            # Apply action
            obs = self._step(action)
            self.steps += 1

            # Determine when is the episode over
            # currently just a timeout after 100 steps
            done = self.steps > int(max_steps)

            # Set info, tell the algorithm the termination was due to a timeout
            # (the episode was truncated)
            info = {"TimeLimit.truncated": self.steps > int(max_steps)}

            return obs, 0., done, info

    # Initialize Environment
    env = ComposedEnv(name="rx", rate=rate, graph=graph, engine=engine)
    env = Flatten(env)
    env = w.rescale_action.RescaleAction(env, min_action=-1.5, max_action=1.5)

    # Initialize model
    # model = sb.SAC.load(LOG_DIR, env, device="cuda", verbose=1, tensorboard_log=LOG_DIR)

    # First train in simulation
    env.render("human")

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done = env.reset(), False
        while not done:
            action = env.action_space.sample()
            # action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            rgb = env.render("rgb_array")
