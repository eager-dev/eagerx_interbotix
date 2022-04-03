import os
os.environ["PYBULLET_EGL"] = "1"
# ^^^^ before importing eagerx_pybullet

# ROS packages required
from eagerx import Object, Bridge, initialize, log, process, Node

# Environment
from eagerx.core.env import EagerxEnv
from eagerx.core.graph import Graph

# Implementation specific
import eagerx.nodes  # Registers butterworth_filter # noqa # pylint: disable=unused-import
import eagerx_pybullet  # Registers PybulletBridge # noqa # pylint: disable=unused-import
import eagerx_interbotix  # Registers objects # noqa # pylint: disable=unused-import
import eagerx_reality  # Registers bridge # noqa # pylint: disable=unused-import

# todo: document pybullet engine nodes, bridge.
# todo: set number of substepss
# todo: set join/link sensors/actuators to separate process (ensure that they are connect to the right physics server)
if __name__ == "__main__":
    initialize("eagerx_core", anonymous=True, log_level=log.INFO)

    # Define rate
    rate = 20.0

    # Initialize empty graph
    graph = Graph.create()

    # Create camera
    # urdf = os.path.dirname(eagerx_interbotix.__file__) + "/camera/assets/realsense2_d435.urdf"
    # cam = Object.make("Camera", "cam", rate=rate, sensors=["rgb"], urdf=urdf, optical_link="camera_color_optical_frame", calibration_link="camera_bottom_screw_frame")
    # graph.add(cam)

    # Create solid object
    # import pybullet_data
    # urdf = "%s/%s.urdf" % (pybullet_data.getDataPath(), "cube_small")
    # cube = Object.make("Solid", "cube", urdf=urdf, rate=rate, sensors=["pos"])
    # graph.add(cube)

    # Create arm
    arm = Object.make("Xseries", "viper", "vx300s", sensors=["pos"], actuators=["pos_control", "gripper_control"],
                      states=["pos", "vel", "gripper"], rate=rate)
    graph.add(arm)

    # Create safety node
    c = arm.config
    collision = dict(workspace="eagerx_interbotix.safety.workspaces/cubes_and_2dof",
                     margin=0.03,
                     gui=False,
                     robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or))
    safe = Node.make("SafetyFilter", "safety", rate, c.joint_names, c.joint_upper, c.joint_lower, c.vel_limit, collision=collision)
    graph.add(safe)

    # Connect the nodes
    graph.connect(action="gripper", target=arm.actuators.gripper_control)
    graph.connect(action="joints", target=safe.inputs.goal)
    graph.connect(source=arm.sensors.pos, target=safe.inputs.current)
    graph.connect(source=safe.outputs.filtered, target=arm.actuators.pos_control)
    graph.connect(source=arm.sensors.pos, observation="joints")

    # Show in the gui
    # graph.gui()

    # Define bridges
    # bridge = Bridge.make("RealBridge", rate=rate, is_reactive=True, process=process.NEW_PROCESS)
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
    env = EagerxEnv(name="rx", rate=rate, graph=graph, bridge=bridge, step_fn=step_fn)

    # First train in simulation
    env.render("human")

    # Evaluate for 30 seconds in simulation
    obs, action = env.reset(), env.action_space.sample()
    action["joints"][:] = arm.config.sleep_positions
    print(f"Steps 0")
    for i in range(int(50000 * rate)):
        if (i+1) % 20 == 0:
            action["joints"][:] = env.action_space["joints"].sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs, action["joints"][:] = env.reset(), env.action_space["joints"].sample()
            # action["joints"][:] = arm.config.sleep_positions
            print(f"Steps {i}")

