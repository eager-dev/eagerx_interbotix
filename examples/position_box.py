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
import stable_baselines3 as sb
from datetime import datetime
import os

NAME = "varyGoal_term_noExcl"
LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"

# todo: increase friction coefficient (seems to glide too much)
# todo: velocity control
# todo: Increase the penalty on velocity
# todo: switch goal with object position
# todo: normalize actions/observations

if __name__ == "__main__":
    eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.WARN)

    # Define rate
    real_reset = False
    rate = 20
    safe_rate = 20
    max_steps = 300

    # Initialize empty graph
    graph = Graph.create()

    # Create camera
    cam = eagerx.Object.make(
        "Camera",
        "cam",
        rate=rate,
        sensors=["rgb"],
        urdf=os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf",
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
    )
    # graph.add(cam)

    # Create solid object
    urdf_path = os.path.dirname(eagerx_interbotix.__file__) + "/solid/assets/"
    solid = eagerx.Object.make(
        "Solid", "solid", urdf=urdf_path + "box.urdf", rate=rate, sensors=["pos"], base_pos=[0, 0, 1], fixed_base=False,
        states=["pos", "vel", "orientation", "angular_vel", "lateral_friction"]
    )
    solid.sensors.pos.space_converter.low = [0, -1, 0]
    solid.sensors.pos.space_converter.high = [1, 1, 0.15]
    solid.states.lateral_friction.space_converter.low = 0.4
    solid.states.lateral_friction.space_converter.high = 0.1
    graph.add(solid)

    # Create solid goal
    goal = eagerx.Object.make(
        "Solid", "goal", urdf=urdf_path + "box_goal.urdf", rate=rate, sensors=["pos"], base_pos=[1, 0, 1], fixed_base=True
    )
    goal.sensors.pos.space_converter.low = [0, -1, 0]
    goal.sensors.pos.space_converter.high = [1, 1, 0.15]
    graph.add(goal)

    # Create arm
    arm = eagerx.Object.make(
        "Xseries",
        "viper",
        "vx300s",
        sensors=["pos", "vel", "ee_pos"],
        actuators=["pos_control"],
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
        reset = eagerx.ResetNode.make("ResetArm", "reset", rate, c.joint_upper, c.joint_lower, gripper=False)
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
    # engine = Engine.make("RealEngine", rate=rate, sync=True, process=process.NEW_PROCESS)
    engine = eagerx.Engine.make("PybulletEngine", rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=0.0)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Set info:
        info = dict()
        # Calculate reward
        ee_pos = obs["ee_position"][0]
        goal = obs["goal"][0]
        can = obs["solid"][0]
        vel = obs["velocity"][0]
        # Penalize distance of the end-effector to the object
        rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - can) - 0.033)
        # Penalize distance of the objec to the goal
        rwd_dist = 3.0 * -np.linalg.norm(goal - can)
        # Penalize actions (indirectly, by punishing the angular velocity.
        rwd_ctrl = 0.1 * -np.square(vel).sum()
        rwd = rwd_dist + rwd_ctrl + rwd_near
        # Determine done flag
        if steps > max_steps:  # Max steps reached
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False | (np.linalg.norm(can[:2]) > 1.0)  # Can is out of reach
            if done:
                rwd = -50
        # done = done | (np.linalg.norm(goal - can) < 0.1 and can[2] < 0.05)  # Can has not fallen down & within threshold.
        return obs, rwd, done, info

    # Define reset function
    def reset_fn(env):
        states = env.state_space.sample()

        # Set orientation
        states["goal/orientation"] = np.array([0, 0, 0, 1])
        states["solid/orientation"] = states["goal/orientation"]

        # Sample new starting state (at least 17 cm from goal)
        z = 0.035
        radius = 0.21
        goal_pos = np.array([0.35, 0, z])
        while True:
            can_pos = np.concatenate(
                [
                    np.random.uniform(low=0, high=0.4 * radius, size=1),
                    np.random.uniform(low=-1.2 * radius, high=1.2 * radius, size=1),
                    [z],
                ]
            )
            if np.linalg.norm(can_pos) > radius:
                break
        states["solid/pos"] = can_pos + goal_pos
        states["goal/pos"] = goal_pos

        # Set gripper to closed position
        states["viper/gripper"][0] = 0
        return states

    # Initialize Environment
    env = EagerxEnv(name="rx", rate=rate, graph=graph, engine=engine, step_fn=step_fn, reset_fn=reset_fn, exclude=[])

    # Initialize model
    os.mkdir(LOG_DIR)
    graph.save(f"{LOG_DIR}/graph.yaml")
    model = sb.SAC("MlpPolicy", Flatten(env), device="cuda", verbose=1, tensorboard_log=LOG_DIR)

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
