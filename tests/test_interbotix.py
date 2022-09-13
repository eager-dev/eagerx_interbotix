import eagerx
import numpy as np
import eagerx
import eagerx_interbotix

# Other
import inspect
import pytest

NP = eagerx.process.NEW_PROCESS
ENV = eagerx.process.ENVIRONMENT

cam_intrinsics = {"image_width": 640,
                  "image_height": 480,
                  "camera_name": "logitech_c170",
                  "camera_matrix": {"rows": 3, "cols": 3,
                                    "data": [744.854391488828, 0, 327.2505862760107, 0, 742.523670731623, 207.0294448122543, 0,
                                             0, 1]},
                  "distortion_model": "plumb_bob",
                  "distortion_coefficients": {
                      "rows": 1,
                      "cols": 5,
                      "data": [0.1164823030284325, -0.7022013182646298, -0.01409335811907957, 0.001216661775149573, 0]},
                  "rectification_matrix": {
                      "rows": 3,
                      "cols": 3,
                      "data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                  "projection_matrix": {
                      "rows": 3,
                      "cols": 4,
                      "data": [742.5888671875, 0, 327.7492634819646, 0, 0, 741.0902709960938, 202.927556117038, 0, 0, 0, 1, 0]}
                  }


def position_control(_graph, _arm, source_goal, safe_rate):
    # Add position control actuator
    if "pos_control" not in _arm.config.actuators:
        _arm.config.actuators.append("pos_control")

    # Create safety node
    from eagerx_interbotix.safety.node import SafePositionControl
    c = _arm.config
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
    _graph.add(safe)

    # Connecting safety filter to arm
    _graph.connect(**source_goal, target=safe.inputs.goal)
    _graph.connect(source=_arm.sensors.position, target=safe.inputs.current)
    _graph.connect(source=safe.outputs.filtered, target=_arm.actuators.pos_control)

    return safe


def velocity_control(_graph, _arm, source_goal, safe_rate):
    # Add velocity control actuator
    if "vel_control" not in _arm.config.actuators:
        _arm.config.actuators.append("vel_control")

    # Create safety node
    from eagerx_interbotix.safety.node import SafeVelocityControl
    c = _arm.config
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
        # workspace="eagerx_interbotix.safety.workspaces/exclude_ground_minus_2m",
        margin=0.01,  # [cm]
        gui=False,
        robot=dict(urdf=c.urdf, basePosition=c.base_pos, baseOrientation=c.base_or),
    )
    safe = SafeVelocityControl.make(
        "safety",
        safe_rate,
        c.joint_names,
        c.joint_upper,
        c.joint_lower,
        [0.2 * vl for vl in c.vel_limit],
        checks=3,
        collision=collision,
    )
    _graph.add(safe)

    # Connecting goal
    _graph.connect(**source_goal, target=safe.inputs.goal)
    # Connecting safety filter to arm
    _graph.connect(source=_arm.sensors.position, target=safe.inputs.position)
    _graph.connect(source=_arm.sensors.velocity, target=safe.inputs.velocity)
    _graph.connect(source=safe.outputs.filtered, target=_arm.actuators.vel_control)

    return safe



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
    rate = 10  # 20
    safe_rate = 20
    T_max = 1.0  # [sec]
    add_bias = True
    excl_z = False
    USE_POS_CONTROL = False

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create camera
    # from eagerx_interbotix.camera.objects import Camera
    # urdf_path = "/".join(inspect.getfile(Camera).split("/")[:-1]) + "/assets/realsense2_d435.urdf"
    # cam = Camera.make(
    #     "cam",
    #     rate=rate,
    #     sensors=["rgb"],
    #     urdf=urdf_path,
    #     optical_link="camera_color_optical_frame",
    #     calibration_link="camera_bottom_screw_frame",
    # )
    # graph.add(cam)

    # Create solid object
    from eagerx_interbotix.solid.solid import Solid
    solid = Solid.make(
        "solid",
        urdf="cube_small.urdf",
        rate=rate,
        cam_translation=[0.811, 0.527, 0.43],
        cam_rotation=[0.321, 0.801, -0.466, -0.197],
        cam_index=2,
        cam_intrinsics=cam_intrinsics,
        sensors=["position", "yaw"],  # select robot_view to render.
        states=["position", "velocity", "orientation", "angular_vel", "lateral_friction"],
    )
    solid.sensors.position.space.update(low=[-1, -1, 0], high=[1, 1, 0.13])
    graph.add(solid)

    # Create solid goal
    from eagerx_interbotix.solid.goal import Goal
    goal = Goal.make(
        "goal",
        urdf="cube_small.urdf",
        rate=rate,
        sensors=["position"],
        states=["position", "orientation"],
    )
    goal.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0.15])
    graph.add(goal)

    # Circular goal
    x, y, z = 0.30, 0.0, 0.05
    dx, dy = 0.1, 0.20
    solid.states.lateral_friction.space.update(low=0.1, high=0.4)
    solid.states.orientation.space.update(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])
    solid.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
    goal.states.orientation.space.update(low=[0, 0, 0, 1], high=[0, 0, 0, 1])
    goal.states.position.space.update(low=[x, y, z], high=[x, y, z])


    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    robot_type = "vx300s"
    arm = Xseries.make(
        name=robot_type,
        robot_type=robot_type,
        sensors=["position", "velocity", "ee_pos", "ee_orn"],
        actuators=[],
        states=["position", "velocity", "gripper"],
        rate=rate,
    )
    arm.states.gripper.space.update(low=[0.], high=[0.])  # Set gripper to closed position
    arm.states.position.space.low[-2] = np.pi / 2
    arm.states.position.space.high[-2] = np.pi / 2
    graph.add(arm)

    # Create IK node
    from eagerx_interbotix.ik.node import EndEffectorDownward
    import eagerx_interbotix.xseries.mr_descriptions as mrd

    robot_des = getattr(mrd, robot_type)
    c = arm.config
    ik = EndEffectorDownward.make("ik",
                                  rate,
                                  c.joint_names,
                                  robot_des.Slist.tolist(),
                                  robot_des.M.tolist(),
                                  c.joint_upper,
                                  c.joint_lower,
                                  max_dxyz=[0.2, 0.2, 0.2],  # 10 cm / sec
                                  max_dyaw=2 * np.pi / 2,  # 1/5 round / second
                                  )
    graph.add(ik)

    if USE_POS_CONTROL:
        safe = position_control(graph, arm, dict(source=ik.outputs.target), safe_rate)
    else:
        safe = velocity_control(graph, arm, dict(source=ik.outputs.dtarget), safe_rate)

    # Connecting observations
    graph.connect(source=arm.sensors.position, observation="joints")
    graph.connect(source=arm.sensors.velocity, observation="velocity")
    graph.connect(source=arm.sensors.ee_pos, observation="ee_position")
    graph.connect(source=solid.sensors.position, observation="solid")
    graph.connect(source=solid.sensors.yaw, observation="yaw")
    graph.connect(source=goal.sensors.position, observation="goal")
    # Connect IK
    graph.connect(source=arm.sensors.position, target=ik.inputs.current)
    graph.connect(source=arm.sensors.ee_pos, target=ik.inputs.xyz)
    graph.connect(source=arm.sensors.ee_orn, target=ik.inputs.orn)
    # Connecting actions
    graph.connect(action="dxyz", target=ik.inputs.dxyz)
    graph.connect(action="dyaw", target=ik.inputs.dyaw)

    # Define engines
    from eagerx_pybullet.engine import PybulletEngine
    engine = PybulletEngine.make(rate=safe_rate, gui=False, egl=False, sync=True, real_time_factor=0.0, process=engine_p)

    # Make backend
    # from eagerx.backends.ros1 import Ros1
    # backend = Ros1.make()
    from eagerx.backends.single_process import SingleProcess
    backend = SingleProcess.make()

    # Define environment
    from eagerx_interbotix.env import ArmEnv

    # Initialize Environment
    env = ArmEnv(name=f"ArmEnv",
                 rate=rate,
                 graph=graph,
                 engine=engine,
                 backend=backend,
                 exclude_z=excl_z,
                 max_steps=int(T_max * rate),
                 add_bias=add_bias)

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
