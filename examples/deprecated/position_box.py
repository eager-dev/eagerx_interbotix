import eagerx
from eagerx.wrappers.flatten import Flatten
import eagerx_interbotix

# Other
import numpy as np
import gym.wrappers as w
import stable_baselines3 as sb
from datetime import datetime
import os

NAME = "refactor_varyGoal_term_noExcl"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # Define rate
    real_reset = False
    rate = 20
    safe_rate = 20
    max_steps = 300

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create camera
    from eagerx_interbotix.camera.objects import Camera
    cam = Camera.make(
        "cam",
        rate=rate,
        sensors=["rgb"],
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
    )
    # graph.add(cam)

    # Create solid object
    from eagerx_interbotix.solid.solid import Solid
    urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    solid = Solid.make(
        "solid", urdf=urdf_path + "box.urdf", rate=rate, sensors=["pos"], base_pos=[0, 0, 1], fixed_base=False,
        states=["pos", "vel", "orientation", "angular_vel", "lateral_friction"]
    )
    x, y, z, dx, dy = 0.35, 0, 0.035, 0.1, 0.20
    solid.sensors.pos.space.update(low=[0, -1, 0], high=[1, 1, z + 0.1])
    solid.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    solid.states.pos.space.update(low=[x, y - dy, z], high=[x + dx, y + dy, z])
    graph.add(solid)

    # Create solid goal
    goal = Solid.make(
        "goal", urdf=urdf_path + "box_goal.urdf", rate=rate, sensors=["pos"], base_pos=[1, 0, 1], fixed_base=True
    )
    goal.sensors.pos.space.update(low=[x, y, z], high=[x, y, z])
    goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    goal.states.pos.space.update(low=[x, y, z], high=[x, y, z])
    graph.add(goal)

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    arm = Xseries.make(
        "viper",
        "vx300s",
        sensors=["pos", "vel", "ee_pos"],
        actuators=["pos_control"],
        states=["pos", "vel", "gripper"],
        rate=rate,
    )
    graph.add(arm)

    # Create safety node
    from eagerx_interbotix.safety.node import SafePositionControl
    c = arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
        # workspace="eagerx_interbotix.safety.workspaces/exclude_ground_minus_2m",
        margin=0.01,  # [cm]
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = SafePositionControl.make(
        "safety",
        safe_rate,
        c.joint_names,
        c.joint_upper,
        c.joint_lower,
        [0.2 * vl for vl in c.vel_limit],
        checks=3,
        collision=collision,
    )
    graph.add(safe)

    # Connecting observations
    graph.connect(source=arm.sensors.pos, observation="joints")
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
    graph.connect(source=arm.sensors.vel, observation="velocity")
    graph.connect(source=solid.sensors.pos, observation="solid")
    graph.connect(source=goal.sensors.pos, observation="goal")
    # Connecting actions
    graph.connect(action="position", target=safe.inputs.goal)
    # Connecting safety filter to arm
    graph.connect(source=arm.sensors.pos, target=safe.inputs.current)
    graph.connect(source=safe.outputs.filtered, target=arm.actuators.pos_control)

    # Create reset node
    if real_reset:
        from eagerx_interbotix.reset.node import ResetArm
        reset = ResetArm.make("reset", rate, c.joint_upper, c.joint_lower, gripper=False)
        graph.add(reset)

        # Disconnect simulation-specific connections
        graph.disconnect(action="joints", target=safe.inputs.goal)

        # Connect target state we are resetting
        graph.connect(source=arm.states.pos, target=reset.targets.goal)
        # Connect actions to feedthrough (that are overwritten during a reset)
        graph.connect(action="position", target=reset.feedthroughs.joints)
        # Connect joint output to safety filter
        graph.connect(source=reset.outputs.joints, target=safe.inputs.goal)
        # Connect inputs to determine reset status
        graph.connect(source=arm.sensors.pos, target=reset.inputs.joints)
        graph.connect(source=safe.outputs.in_collision, target=reset.inputs.in_collision, skip=True)

    # Show in the gui
    # graph.gui()

    # Define engines
    # from eagerx_reality.engine import RealEngine
    # engine = RealEngine.make(rate=rate, sync=True, process=eagerx.NEW_PROCESS)
    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

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
            ee_pos = obs["ee_position"][0]
            goal = obs["goal"][0]
            can = obs["solid"][0]
            vel = obs["velocity"][0]
            # Penalize distance of the end-effector to the object
            rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - can) - 0.033)
            # Penalize distance of the object to the goal
            rwd_dist = 2.0 * -np.linalg.norm(goal - can)
            # Penalize actions (indirectly, by punishing the angular velocity.
            rwd_ctrl = 0.1 * -np.linalg.norm(vel)
            rwd = rwd_dist + rwd_ctrl + rwd_near
            # Print rwd build-up
            # msg = f"rwd={rwd: .2f} | near={100*rwd_near/rwd: .1f} | dist={100*rwd_dist/rwd: .1f} | ctrl={100*rwd_ctrl/rwd: .1f}"
            # print(msg)
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
                solid_pos = self.state_space["solid/pos"].sample()
                goal_pos = self.state_space["goal/pos"].sample()
                if np.linalg.norm(solid_pos[:2] - goal_pos[:2]) > radius:
                    states["solid/pos"] = solid_pos
                    break

            # Perform reset
            obs = self._reset(states)
            return obs


    # Initialize Environment
    env = ArmEnv(name="ArmEnv", rate=rate, graph=graph, engine=engine, backend=backend, force_start=True, max_steps=max_steps)
    sb_env = Flatten(env)
    sb_env = w.rescale_action.RescaleAction(sb_env, min_action=-1.5, max_action=1.5)

    # Initialize model
    os.mkdir(LOG_DIR)
    graph.save(f"{LOG_DIR}/graph.yaml")
    model = sb.SAC("MlpPolicy", sb_env, device="cuda", verbose=1, tensorboard_log=LOG_DIR)

    # Create experiment directory
    delta_steps = 100000
    for i in range(1, 30):
        model.learn(delta_steps)
        model.save(f"{LOG_DIR}/model_{i*delta_steps}")

    # First train in simulation
    env.render("human")

    # Evaluate
    for eps in range(5000):
        print(f"Episode {eps}")
        _, done = env.reset(), False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rgb = env.render("rgb_array")
