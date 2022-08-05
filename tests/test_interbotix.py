import eagerx

# Other
import inspect
import pytest

NP = eagerx.process.NEW_PROCESS
ENV = eagerx.process.ENVIRONMENT


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "eps, num_steps, sync, rtf, p",
    [(3, 20, True, 0, NP), (3, 20, True, 0, ENV)]
)
def test_interbotix(eps, num_steps, sync, rtf, p):
    eagerx.set_log_level(eagerx.WARN)

    # Define unique name for test environment
    name = f"{eps}_{num_steps}_{sync}_{p}"
    engine_p = p

    # Define rate
    rate = 5

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create camera
    from eagerx_interbotix.camera.objects import Camera
    urdf_path = "/".join(inspect.getfile(Camera).split("/")[:-1]) + "/assets/realsense2_d435.urdf"
    cam = Camera.make(
        "cam",
        rate=rate,
        sensors=["rgb"],
        urdf=urdf_path,
        optical_link="camera_color_optical_frame",
        calibration_link="camera_bottom_screw_frame",
    )
    graph.add(cam)

    # Create solid object
    from eagerx_interbotix.solid.solid import Solid
    cube = Solid.make("cube", urdf="cube_small.urdf", rate=rate, sensors=["pos"])
    graph.add(cube)

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    arm = Xseries.make(
        "viper",
        "vx300s",
        sensors=["pos"],
        actuators=["pos_control", "gripper_control"],
        states=["pos", "vel", "gripper"],
        rate=rate,
    )
    graph.add(arm)

    # Create safety node
    from eagerx_interbotix.safety.node import SafePositionControl
    c = arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_behind_left_workspace",
        margin=0.02,
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = SafePositionControl.make("safety", 20, c.joint_names, c.joint_upper, c.joint_lower, c.vel_limit, checks=5, collision=collision)
    graph.add(safe)

    # Create reset node
    from eagerx_interbotix.reset.node import ResetArm
    reset = ResetArm.make("reset", 5, c.joint_upper, c.joint_lower, gripper=True)
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

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=20, gui=False, egl=False, sync=True, real_time_factor=0, process=engine_p)

    # Make backend
    # from eagerx.backends.ros1 import Ros1
    # backend = Ros1.make()
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Define environment
    class TestEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start):
            self.steps = 0
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start)

        def step(self, action):
            obs = self._step(action)
            # Determine when is the episode over
            self.steps += 1
            done = self.steps > 500
            return obs, 0, done, {}

        def reset(self):
            # Reset steps counter
            self.steps = 0

            # Sample states
            states = self.state_space.sample()

            # Perform reset
            obs = self._reset(states)
            return obs

    # Initialize Environment
    env = TestEnv(name=name, rate=rate, graph=graph, engine=engine, backend=backend, force_start=True)

    # Evaluate for 30 seconds in simulation
    _, action = env.reset(), env.action_space.sample()
    for i in range(3):
        obs, reward, done, info = env.step(action)
        if done:
            _, action = env.reset(), env.action_space.sample()
            _rgb = env.render("rgb_array")
            print(f"Episode {i}")
    print("\n[Finished]")

    # Shutdown
    env.shutdown()
    print("\nShutdown")


if __name__ == "__main__":
    test_interbotix(3, 20, True, 0, NP)
    test_interbotix(3, 20, True, 0, ENV)
