import eagerx
from eagerx.wrappers.flatten import Flatten
import eagerx_interbotix

# Other
import yaml
import numpy as np
import gym.wrappers as w
import stable_baselines3 as sb
import os

# NAME = "space_box_dynamicsRandomization_2022-07-22-1313"
# NAME = "first_real_experiment_2022-08-10-1428"
NAME = "safety_bias_2022-08-11-1218"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}"
GRAPH_FILE = f"graph.yaml"

if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Camera settings
    cam_index_robot_view = 2
    cam_index_overview = 0
    CAM_PATH = "/home/r2ci/eagerx-dev/eagerx_interbotix/assets/calibrations"
    # CAM_INTRINSICS = "logitech_c922.yaml"
    CAM_INTRINSICS = "logitech_camera.yaml"
    # CAM_EXTRINSICS = "2022-08-10-1413_logitech_camera_vx300s_inaccurate_0_029error.yaml"
    # CAM_EXTRINSICS = "eye_hand_calibration_2022-08-10-1726.yaml"
    # CAM_EXTRINSICS = "eye_hand_calibration_2022-08-10-1757.yaml"
    CAM_EXTRINSICS = "extrinsic.yaml"
    with open(f"{CAM_PATH}/{CAM_INTRINSICS}", "r") as f:
        cam_intrinsics = yaml.safe_load(f)
    with open(f"{CAM_PATH}/{CAM_EXTRINSICS}", "r") as f:
        cam_extrinsics = yaml.safe_load(f)
    cam_translation = cam_extrinsics["camera_to_robot"]["translation"]
    cam_rotation = cam_extrinsics["camera_to_robot"]["rotation"]

    must_render = False
    render_rate = 5
    max_steps = 300
    rate = 20
    safe_rate = 20

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
    solid.config.cam_translation = cam_translation
    solid.config.cam_rotation = cam_rotation
    solid.config.cam_intrinsics = cam_intrinsics

    # Add rendering
    if must_render:
        solid.config.sensors = ["position", "yaw", "robot_view"]
        solid.config.cam_index = cam_index_robot_view

        # Create camera
        from eagerx_interbotix.camera.objects import Camera

        # todo: set correct cam overview location
        cam = Camera.make(
            "cam",
            rate=render_rate,
            sensors=["image"],
            urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
            optical_link="camera_color_optical_frame",
            calibration_link="camera_bottom_screw_frame",
            camera_index=cam_index_overview,
        )
        graph.add(cam)
        # Create overlay
        from eagerx_interbotix.overlay.node import Overlay
        overlay = Overlay.make("overlay", rate=render_rate, resolution=[480, 480], caption="robot view")
        graph.add(overlay)
        # Connect
        graph.connect(source=solid.sensors.robot_view, target=overlay.inputs.main)
        graph.connect(source=cam.sensors.image, target=overlay.inputs.thumbnail)
        graph.render(source=overlay.outputs.image, rate=render_rate, encoding="bgr")

    # Define engines
    from eagerx_reality.engine import RealEngine
    engine = RealEngine.make(rate=rate, sync=False, process=eagerx.NEW_PROCESS)
    # from eagerx_pybullet.engine import PybulletEngine
    # engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Make backend
    from eagerx.backends.ros1 import Ros1
    backend = Ros1.make()
    # from eagerx.backends.single_process import SingleProcess
    # backend = SingleProcess.make()

    # Define environment
    class ArmEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, max_steps: int):
            self.steps = 0
            self.max_steps = max_steps
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start)

        def step(self, action):
            # Step the environment
            self.steps += 1
            info = dict()
            obs = self._step(action)

            # Calculate reward
            yaw = obs["yaw"]
            ee_pos = obs["ee_position"][0]
            goal = obs["goal"][0]
            can = obs["solid"][0]
            vel = obs["velocity"][0]
            des_vel = action["velocity"][0] if "velocity" in action else 0*vel
            # Penalize distance of the end-effector to the object
            rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - can) - 0.05)
            # Penalize distance of the object to the goal
            rwd_dist = 2.0 * -np.linalg.norm(goal - can)
            # Penalize actions (indirectly, by punishing the angular velocity.
            rwd_ctrl = 0.1 * -np.linalg.norm(des_vel - vel)
            rwd = rwd_dist + rwd_ctrl + rwd_near
            # Print solid location
            # msg = f"yaw={yaw} | pos={str(can.round(2))}"
            # print(msg)
            # Print rwd build-up
            # msg = f"rwd={rwd: .2f} | near={100*rwd_near/rwd: .1f} | dist={100*rwd_dist/rwd: .1f} | ctrl={100*rwd_ctrl/rwd: .1f}"
            # Determine done flag
            if self.steps > self.max_steps:  # Max steps reached
                done = True
                info["TimeLimit.truncated"] = True
            else:
                done = False | (np.linalg.norm(can[:2]) > 1.0)  # Can is out of reach
                if done:
                    rwd = -50
            # done = done | (np.linalg.norm(goal - can) < 0.1 and can[2] < 0.05)  # Can has not fallen down & within threshold.
            return obs, rwd, done, info

        def reset(self):
            # Reset steps counter
            self.steps = 0

            # Sample states
            states = self.state_space.sample()

            # Sample new starting positions (at least 17 cm from goal)
            radius = 0.17
            while True:
                solid_pos = self.state_space["solid/position"].sample()
                goal_pos = self.state_space["goal/position"].sample()
                if np.linalg.norm(solid_pos[:2] - goal_pos[:2]) > radius:
                    states["solid/position"] = solid_pos
                    break

            # Set initial position state
            if "solid/aruco/position" in states:
                self.state_space["solid/aruco/position"] = solid_pos

            # Perform reset
            obs = self._reset(states)
            return obs

    # Initialize env
    env = ArmEnv(name="ArmEnv", rate=rate, graph=graph, engine=engine, backend=backend, force_start=True,
                 max_steps=max_steps)
    env.render()
    sb_env = Flatten(env)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.5, max_action=1.5)

    # Load model
    model = sb.SAC.load(f"{LOG_DIR}/{MODEL_NAME}", sb_env, device="cuda", verbose=1)

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        obs, done = sb_env.reset(), False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sb_env.step(action)
