# EAGERx imports
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletBridge # noqa # pylint: disable=unused-import
import eagerx_interbotix  # Registers objects # noqa # pylint: disable=unused-import

# Other
import os
import pytest

NP = eagerx.process.NEW_PROCESS
ENV = eagerx.process.ENVIRONMENT


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "eps, num_steps, is_reactive, rtf, p",
    [(3, 20, True, 0, NP), (3, 20, True, 0, ENV)]
)
def test_interbotix(eps, num_steps, is_reactive, rtf, p):
    # Start roscore
    roscore = eagerx.initialize("eagerx_core", anonymous=True, log_level=eagerx.log.WARN)

    # Define unique name for test environment
    name = f"{eps}_{num_steps}_{is_reactive}_{p}"
    bridge_p = p

    # Define rate
    rate = 5

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
    graph.add(cam)

    # Create solid object
    cube = eagerx.Object.make("Solid", "cube", urdf="cube_small.urdf", rate=rate, sensors=["pos"])
    graph.add(cube)

    # Create arm
    arm = eagerx.Object.make(
        "Xseries",
        "viper",
        "vx300s",
        sensors=["pos"],
        actuators=["pos_control", "gripper_control"],
        states=["pos", "vel", "gripper"],
        rate=rate,
    )
    graph.add(arm)

    # Create safety node
    c = arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_behind_left_workspace",
        margin=0.02,
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = eagerx.Node.make(
        "SafetyFilter", "safety", 20, c.joint_names, c.joint_upper, c.joint_lower, c.vel_limit, checks=5, collision=collision
    )
    graph.add(safe)

    # Create reset node
    reset = eagerx.ResetNode.make("ResetArm", "reset", 5, c.joint_upper, c.joint_lower, gripper=True)
    graph.add(reset)

    # Connect the nodes
    graph.connect(source=arm.states.pos, target=reset.targets.goal)
    graph.connect(action="gripper", target=reset.feedthroughs.gripper)
    graph.connect(source=reset.outputs.gripper, target=arm.actuators.gripper_control)
    graph.connect(action="joints", target=reset.feedthroughs.joints)
    graph.connect(source=reset.outputs.joints, target=safe.inputs.goal)
    graph.connect(source=arm.sensors.pos, target=safe.inputs.current)
    graph.connect(source=arm.sensors.pos, target=reset.inputs.joints)
    graph.connect(source=safe.outputs.filtered, target=arm.actuators.pos_control)
    graph.connect(source=safe.outputs.in_collision, target=reset.inputs.in_collision, skip=True)
    graph.connect(source=arm.sensors.pos, observation="joints")

    # Define bridges
    bridge = eagerx.Bridge.make("PybulletBridge", rate=20, gui=False, egl=False, is_reactive=True, real_time_factor=0,
                                process=bridge_p)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > num_steps
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    env = EagerxEnv(name=name, rate=rate, graph=graph, bridge=bridge, step_fn=step_fn)

    # Evaluate
    for e in range(eps):
        print(f"Episode {e}")
        _, done = env.reset(), False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rgb = env.render("rgb_array")

    # Shutdown
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("\nShutdown")
