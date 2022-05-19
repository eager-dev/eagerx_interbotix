# EAGERx imports
from eagerx.wrappers.flatten import Flatten
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletEngine # noqa # pylint: disable=unused-import
import eagerx_interbotix  # Registers objects # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers engine # noqa # pylint: disable=unused-import

# Other
import numpy as np
import gym.wrappers as w
import os

if __name__ == "__main__":
    eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.WARN)

    # Define rate
    real_reset = True
    rate = 20
    safe_rate = 20
    max_steps = rate * 15

    # Initialize empty graph
    graph = Graph.create()

    # Get camera index
    # import cv2
    # def returnCameraIndexes():
    #     # checks the first 10 indexes.
    #     index = 0
    #     arr = []
    #     i = 10
    #     while i > 0:
    #         cap = cv2.VideoCapture(index)
    #         if cap.read()[0]:
    #             arr.append(index)
    #             cap.release()
    #         index += 1
    #         i -= 1
    #     return arr

    # Create camera
    cam = eagerx.Object.make(
        "Camera",
        "cam",
        rate=rate,
        sensors=["rgb"],
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
        camera_index=0,
        # camera_index=returnCameraIndexes()[-1],
    )
    graph.add(cam)
    graph.render(cam.sensors.rgb, rate=rate)

    # # Create solid object
    # urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    # solid = eagerx.Object.make(
    #     "Solid", "solid", urdf=urdf_path + "can.urdf", rate=rate, sensors=["pos"], base_pos=[0, 0, 1], fixed_base=False,
    #     states=["pos", "vel", "orientation", "angular_vel", "lateral_friction"]
    # )
    # solid.sensors.pos.space_converter.low = [0, -1, 0]
    # solid.sensors.pos.space_converter.high = [1, 1, 0.15]
    # solid.states.lateral_friction.space_converter.low = 0.4
    # solid.states.lateral_friction.space_converter.high = 0.1
    # graph.add(solid)
    # graph.connect(source=solid.sensors.pos, observation="solid")

    # Create arm
    arm = eagerx.Object.make(
        "Xseries",
        "px150",
        "px150",
        sensors=["pos"],
        actuators=["pos_control", "gripper_control"],
        states=["pos", "vel", "gripper"],
        rate=rate,
    )
    graph.add(arm)

    # Create safety node
    c = arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
        # workspace="eagerx_interbotix.safety.workspaces/exclude_ground_minus_2m",
        margin=0.01,  # [cm]
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = eagerx.Node.make(
        "SafePositionControl",
        "safety",
        safe_rate,
        c.joint_names,
        c.joint_upper,
        c.joint_lower,
        [0.5 * vl for vl in c.vel_limit],
        checks=5,
        collision=collision,
    )
    graph.add(safe)

    # Connecting observations
    graph.connect(source=arm.sensors.pos, observation="joints")
    # Connecting actions
    graph.connect(action="joints", target=arm.actuators.pos_control)
    graph.connect(action="joints", target=safe.inputs.goal)
    graph.connect(action="gripper", target=arm.actuators.gripper_control)
    # Connecting safety filter to arm
    graph.connect(source=arm.sensors.pos, target=safe.inputs.current)
    # graph.connect(source=safe.outputs.filtered, target=arm.actuators.pos_control)

    # Create reset node
    if real_reset:
        reset = eagerx.ResetNode.make("ResetArm", "reset", rate, c.joint_upper, c.joint_lower, gripper=True)
        graph.add(reset)

        # Disconnect simulation-specific connections
        graph.disconnect(action="joints", target=arm.actuators.pos_control)
        graph.disconnect(action="joints", target=safe.inputs.goal)
        graph.disconnect(action="gripper", target=arm.actuators.gripper_control)

        # Connect target state we are resetting
        graph.connect(source=arm.states.pos, target=reset.targets.goal)
        # Connect actions to feedthrough (that are overwritten during a reset)
        graph.connect(action="gripper", target=reset.feedthroughs.gripper)
        graph.connect(action="joints", target=reset.feedthroughs.joints)
        # Connect joint output to safety filter
        graph.connect(source=reset.outputs.joints, target=arm.actuators.pos_control)
        graph.connect(source=reset.outputs.joints, target=safe.inputs.goal)
        graph.connect(source=reset.outputs.gripper, target=arm.actuators.gripper_control)
        # Connect inputs to determine reset status
        graph.connect(source=arm.sensors.pos, target=reset.inputs.joints)
        graph.connect(source=safe.outputs.in_collision, target=reset.inputs.in_collision, skip=True)

    # Define engines
    # graph.gui()
    real = True
    if real:
        engine = eagerx.Engine.make("RealEngine", rate=rate, sync=True, process=eagerx.process.NEW_PROCESS)
    else:
        engine = eagerx.Engine.make("PybulletEngine", rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    import gym
    from typing import Tuple, Dict

    class ComposedEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, force_start=True, max_steps=200):
            super(ComposedEnv, self).__init__(name, rate, graph, engine, force_start=force_start)
            self.steps = None
            self.max_steps = max_steps

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

            # Set camera position
            if "cam/pos" in states:
                states["cam/pos"] = np.array([0.6, 0.1, 0.4], dtype="float32")
            if "cam/orientation" in states:
                states["cam/orientation"] = np.array([0.377, -0.04, -0.92,  0.088], dtype="float32")

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
            info = {"TimeLimit.truncated": self.steps > int(self.max_steps)}

            return obs, 0., done, info

    # Initialize Environment
    env = ComposedEnv(name="rx", rate=rate, graph=graph, engine=engine)
    env = Flatten(env)

    # First train in simulation
    env.render("human")

    # Evaluate
    t = 2.5
    waypoints = [
        [1.0, -1.0, 0.2, 0., 1.1, 0.7],
        [1.0, 1.0, 0.2, 0., 1.1, 0.7],
        [1.0, -1.0, 0.2, 0., 1.1, 0.7],
        [1.0, 1.0, 0.2, 0., 1.1, 0.7],
        [1.0, -1.0, 0.2, 0., 1.1, 0.7],
        [1.0, 1.0, 0.2, 0., 1.1, 0.7],
        [1.0, -1.0, 0.2, 0., 1.1, 0.7],
        [1.0, 1.0, 0.2, 0., 1.1, 0.7],
        [1.0, -1.0, 0.2, 0., 1.1, 0.7],
        [1.0, 1.0, 0.2, 0., 1.1, 0.7],
    ]
    from eagerx_interbotix.utils import save_frames_as_gif
    RUN = "STOP"
    PATH_ANIMATION = "/home/r2ci/sim2real"
    for eps in range(5):
        print(f"Episode {eps}")
        obs, done = env.reset(), False
        i = 0
        frames = []
        FILE_ANIMATION = f"{RUN}_{real}_eps{eps}.gif"
        while not done:
            wp_idx = i // int(t * rate)
            if wp_idx > len(waypoints):
                action = waypoints[-1]
            else:
                action = waypoints[wp_idx]
            i += 1
            obs, reward, done, info = env.step(action)
            rgb = env.render("rgb_array")
            frames.append(rgb)
        save_frames_as_gif(1/rate, frames, dpi=72, path=PATH_ANIMATION, filename=FILE_ANIMATION)

    sleep_position = [0, -1.80, 1.55, 0.8, 0, 0]