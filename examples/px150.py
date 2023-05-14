import os
import typing as t
from datetime import datetime
import numpy as np
import gym
import gym.wrappers as w

import eagerx
from eagerx.wrappers.flatten import Flatten
import eagerx_interbotix


# Define environment
class ArmEnv(eagerx.BaseEnv):
    def __init__(self, name, rate, graph, engine, backend, max_steps: int):
        self.steps = 0
        self.max_steps = max_steps
        super().__init__(name, rate, graph, engine, backend=backend, force_start=False)

    @property
    def observation_space(self) -> gym.spaces.Space:
        obs_space = self._observation_space
        return obs_space

    def step(self, action):
        # Step the environment
        self.steps += 1
        info = dict()
        rwd = 0.0
        obs = self._step(action)

        # Determine done flag
        done = False if self.steps < self.max_steps else True
        return obs, rwd, done, info

    def reset(self, states: t.Optional[t.Dict[str, np.ndarray]] = None):
        # Reset steps counter
        self.steps = 0

        # Sample states
        _states = self.state_space.sample()

        # Overwrite with user provided states
        if states is not None:
            for key, value in states:
                if key in _states and value.shape == _states[key].shape:
                    _states[key] = value
                else:
                    self.backend.logwarn(f"State `{key}` incorrectly specified.")

        # Perform reset
        obs = self._reset(_states)
        return obs


def position_control(_graph, _arm, source_goal, safe_rate):
    # Add position control actuator
    if "pos_control" not in _arm.config.actuators:
        _arm.config.actuators.append("pos_control")

    # Create safety node
    from eagerx_interbotix.safety.node import SafePositionControl
    c = _arm.config
    # NOTE: The workspace defines the area where the arm is allowed to move.
    collision = dict(
        workspace="eagerx_interbotix.safety.workspaces/exclude_ground",
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
    _graph.connect(source=safe.outputs.filtered, target=arm.actuators.vel_control)

    return safe


if __name__ == "__main__":
    eagerx.set_log_level(eagerx.WARN)

    # NOTE!
    #  - The arm is far from perfect, so you may need to recalibrate the motor gains, and control rates to run smoothly.
    #  - Copilot: https://github.com/bheijden/interbotix_copilot
    #  - Robot description: https://www.trossenrobotics.com/docs/interbotix_xsarms/specifications/px150.html
    #  - Debug/recalibrate motors: https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/
    #  - Settings motors: https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/

    # Define rate
    rate = 10  # 20
    safe_rate = 20
    T_max = 10.0  # [sec]
    USE_POS_CONTROL = False
    MUST_TEST = False

    # Define logging
    NAME = "px150"
    LOG_DIR = os.path.dirname(eagerx_interbotix.__file__) + f"/../logs/{NAME}_{datetime.today().strftime('%Y-%m-%d-%H%M')}"
    MUST_LOG = False

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Create arm
    from eagerx_interbotix.xseries.xseries import Xseries
    robot_type = "px150"
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
                                  max_dxyz=[0.2, 0.2, 0.2],  # [m/s]
                                  max_dyaw=2 * np.pi / 2,  # [rad/s]
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
    # Connect IK
    graph.connect(source=arm.sensors.position, target=ik.inputs.current)
    graph.connect(source=arm.sensors.ee_pos, target=ik.inputs.xyz)
    graph.connect(source=arm.sensors.ee_orn, target=ik.inputs.orn)
    # Connecting actions
    graph.connect(action="dxyz", target=ik.inputs.dxyz)
    graph.connect(action="dyaw", target=ik.inputs.dyaw)

    # Initialize backend
    from eagerx.backends.ros1 import Ros1
    backend = Ros1.make()

    # Define engines
    # from eagerx_pybullet.engine import PybulletEngine
    # engine = PybulletEngine.make(rate=safe_rate, gui=True, egl=True, sync=True, real_time_factor=1.0)
    from eagerx_reality.engine import RealEngine
    engine = RealEngine.make(rate=safe_rate, sync=True)

    # Define environment
    env = ArmEnv(name=f"ArmEnv", rate=rate, graph=graph, engine=engine, backend=backend, max_steps=int(T_max * rate))
    env = Flatten(env)
    env = w.rescale_action.RescaleAction(env, min_action=-1.0, max_action=1.0)

    # Initialize model
    if MUST_LOG:
        os.mkdir(LOG_DIR)
        graph.save(f"{LOG_DIR}/graph.yaml")
    else:
        LOG_DIR = None
        checkpoint_callback = None

    # Evaluate
    eps = 0
    while True:
        eps += 1
        print(f"Episode {eps}")
        _, done = env.reset(), False
        while not done:
            action = env.action_space.sample()*0
            obs, reward, done, info = env.step(action)

