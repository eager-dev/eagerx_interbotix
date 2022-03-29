# ROS packages required
from eagerx import Object, Bridge, initialize, log, process

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletBridge # noqa # pylint: disable=unused-import
import eagerx_interbotix  # Registers objects # noqa # pylint: disable=unused-import

# OTHER
import os
import pytest

NP = process.NEW_PROCESS
ENV = process.ENVIRONMENT

@pytest.mark.parametrize(
    "eps, num_steps, is_reactive, rtf, p",
    [(3, 500, True, 0, NP), (3, 500, True, 0, ENV)]
)
def test_interbotix(eps, num_steps, is_reactive, rtf, p):
    # Start roscore
    roscore = initialize("eagerx_core", anonymous=True, log_level=log.WARN)

    # Define unique name for test environment
    name = f"{eps}_{num_steps}_{is_reactive}_{p}"
    bridge_p = p
    rate = 30

    # Initialize empty graph
    graph = Graph.create()

    # Create camera
    urdf = os.path.dirname(eagerx_interbotix.__file__)
    urdf += "/camera/assets/realsense2_d435.urdf"
    cam = Object.make("Camera", "cam", rate=rate, sensors=["rgb"], urdf=urdf, optical_link="camera_color_optical_frame", calibration_link="camera_bottom_screw_frame")
    graph.add(cam)

    # Create solid object
    import pybullet_data
    urdf = "%s/%s.urdf" % (pybullet_data.getDataPath(), "cube_small")
    cube = Object.make("Solid", "cube", urdf=urdf, rate=rate, sensors=["pos"])
    graph.add(cube)

    # Create arm
    arm = Object.make("Vx300s", "viper", sensors=["pos"], actuators=["pos_control", "gripper_control"],
                      states=["pos", "vel", "gripper"], rate=rate)
    graph.add(arm)

    # Connect the nodes
    graph.connect(action="joints", target=arm.actuators.pos_control)
    graph.connect(action="gripper", target=arm.actuators.gripper_control)
    graph.connect(source=arm.sensors.pos, observation="observation")

    # Define bridges
    bridge = Bridge.make("PybulletBridge", rate=rate, gui=False, is_reactive=is_reactive, real_time_factor=rtf, process=bridge_p)

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

    # Evaluate for 30 seconds in simulation
    _, action = env.reset(), env.action_space.sample()
    for i in range(eps):
        obs, reward, done, info = env.step(action)
        if done:
            _, action = env.reset(), env.action_space.sample()
            print(f"Episode {i}")
    print("\n[Finished]")
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("\n[Shutdown]")
