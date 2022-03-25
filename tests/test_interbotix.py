# ROS packages required
from eagerx import Object, Bridge, initialize, log, process

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten


# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletBridge # noqa # pylint: disable=unused-import
import eagerx_interbotix.vx300s  # Registers Mops # noqa # pylint: disable=unused-import

import pytest

NP = process.NEW_PROCESS
ENV = process.ENVIRONMENT

@pytest.mark.timeout(20)
@pytest.mark.parametrize(
    "eps, is_reactive, rtf, p",
    [(2, True, 0, ENV), (3, True, 0, NP)],
)
def test_integration_interbotix(eps, is_reactive, rtf, p):
    roscore = initialize("eagerx_core", anonymous=True, log_level=log.INFO)

    # Define rate
    rate = 60.0

    # Initialize empty graph
    graph = Graph.create()

    # Create mops
    arm = Object.make("Vx300s", "viper", sensors=["pos"], actuators=["pos_control", "gripper_control"],
                      states=["pos", "vel", "gripper"], rate=rate)
    graph.add(arm)

    # Connect the nodes
    graph.connect(action="joints", target=arm.actuators.pos_control)
    graph.connect(action="gripper", target=arm.actuators.gripper_control)
    graph.connect(source=arm.sensors.pos, observation="observation")

    # Define bridges
    bridge = Bridge.make("PybulletBridge", rate=rate, gui=False, is_reactive=is_reactive, real_time_factor=rtf,
                         process=p)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > 100
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    name = f"interbotix_{eps}_{is_reactive}_{p}"
    env = Flatten(EagerxEnv(name=name, rate=rate, graph=graph, bridge=bridge, step_fn=step_fn))

    # First train in simulation
    env.render("human")

    # First reset
    done, _ = False, env.reset()

    # Run for several episodes
    for j in range(eps):
        print("\n[Episode %s]" % j)
        iter = 0
        while not done:  # and iter < 10:
            iter += 1
            action = env.action_space.sample()
            _obs, _reward, done, _info = env.step(action)
        _obs = env.reset()
        done = False
    print("\n[Finished]")

    # Shutdown
    env.shutdown()
    if roscore:
        roscore.shutdown()
    print("\n[Shutdown]")
