# ROS packages required
from eagerx import Object, Bridge, initialize, log, process

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph
from eagerx.wrappers import Flatten

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletBridge # noqa # pylint: disable=unused-import
import eagerx_interbotix.vx300s  # Registers object # noqa # pylint: disable=unused-import


if __name__ == "__main__":
    initialize("eagerx_core", anonymous=True, log_level=log.INFO)

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

    # Show in the gui
    # graph.gui()

    # Define bridges
    bridge = Bridge.make("PybulletBridge", rate=rate, gui=True, is_reactive=True, real_time_factor=0,
                         process=process.NEW_PROCESS)

    # Define step function
    def step_fn(prev_obs, obs, action, steps):
        # Calculate reward
        rwd = 0
        # Determine done flag
        done = steps > 500
        # Set info:
        info = dict()
        return obs, rwd, done, info

    # Initialize Environment
    env = Flatten(EagerxEnv(name="rx", rate=rate, graph=graph, bridge=bridge, step_fn=step_fn))

    # First train in simulation
    env.render("human")

    # Evaluate for 30 seconds in simulation
    obs = env.reset()
    eps = 0
    action = env.action_space.sample()
    print(f"Episode {eps}")
    for i in range(int(50000 * rate)):
        obs, reward, done, info = env.step(action)
        if i % 500 == 0:
            eps += 1
            obs = env.reset()
            print(f"Episode {eps}")

