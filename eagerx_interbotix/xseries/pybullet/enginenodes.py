from typing import Optional, List, Dict
import pybullet

# IMPORT EAGERX
from eagerx import Space
from eagerx.core.specs import NodeSpec
from eagerx.core.constants import process as p
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class JointController(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = p.ENGINE,
        color: Optional[str] = "green",
        mode: str = "position_control",
        vel_target: List[float] = None,
        pos_gain: List[float] = None,
        vel_gain: List[float] = None,
        max_vel: List[float] = None,
        max_force: List[float] = None,
        delay_state: bool = False,
    ):
        """A spec to create a JointController node that controls a set of joints.

        For more info on `vel_target`, `pos_gain`, and `vel_gain`, see `setJointMotorControlMultiDofArray` in
        https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

        :param name: User specified node name.
        :param rate: Rate (Hz) at which the callback is called.
        :param joints: List of controlled joints. Its order determines the ordering of the applied commands.
        :param process: Process in which this node is launched. See :class:`~eagerx.core.constants.process` for all options.
        :param color: Specifies the color of logged messages & node color in the GUI.
        :param mode: Available: `position_control`, `velocity_control`, `pd_control`, and `torque_control`.
        :param vel_target: The desired velocity. Ordering according to `joints`.
        :param pos_gain: Position gain. Ordering according to `joints`.
        :param vel_gain: Velocity gain. Ordering according to `joints`.
        :param max_vel: in `position_control` this limits the velocity to a maximum.
        :param max_force: Maximum force when mode in [`position_control`, `velocity_control`, `pd_control`]. Ordering
                          according to `joints`.
        :return: NodeSpec
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.inputs = ["tick", "action"]
        spec.config.outputs = ["action_applied"]
        spec.config.states = ["delay"] if delay_state else []

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.joints = joints
        spec.config.mode = mode
        spec.config.vel_target = vel_target if vel_target else [0.0] * len(joints)
        spec.config.pos_gain = pos_gain if pos_gain else [0.2] * len(joints)
        spec.config.vel_gain = vel_gain if vel_gain else [0.2] * len(joints)
        spec.config.max_vel = max_vel if max_vel else [3.14] * len(joints)
        spec.config.max_force = max_force if max_force else [999.0] * len(joints)
        return spec

    def initialize(self, spec: NodeSpec, simulator: Dict):
        """Initializes the joint controller node according to the spec."""
        # We will probably use simulator in callback & reset.
        assert self.process == p.ENGINE, (
            "Simulation node requires a reference to the simulator," " hence it must be launched in the Engine process"
        )
        self.joints = spec.config.joints
        self.mode = spec.config.mode
        self.vel_target = spec.config.vel_target
        self.pos_gain = spec.config.pos_gain
        self.vel_gain = spec.config.vel_gain
        self.max_vel = spec.config.max_vel
        self.max_force = spec.config.max_force
        self.robot = simulator["object"]
        self._p = simulator["client"]
        self.physics_client_id = self._p._client

        self.bodyUniqueId = []
        self.jointIndices = []
        for _idx, pb_name in enumerate(spec.config.joints):
            bodyid, jointindex = self.robot.jdict[pb_name].get_bodyid_jointindex()
            self.bodyUniqueId.append(bodyid), self.jointIndices.append(jointindex)

        self.joint_cb = self._joint_control(
            self._p,
            self.mode,
            self.bodyUniqueId[0],
            self.jointIndices,
            self.pos_gain,
            self.vel_gain,
            self.vel_target,
            self.max_vel,
            self.max_force,
        )

    @register.states(delay=Space(shape=(), low=0.0, high=0.0, dtype="float32"))
    def reset(self, delay=None):
        """Set delay if needed."""
        if delay is not None:
            self.set_delay(delay, "inputs", "action")

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(
        self,
        t_n: float,
        tick: Optional[Msg] = None,
        action: Optional[Msg] = None,
    ):
        """Sets the most recently received `action` in the pybullet joint controller.

        The action is set at the specified rate * real_time_factor.

        The output `action_applied` is the action that was set. If the input `action` comes in at a higher rate than
        this node's rate, `action_applied` may be differnt as only the most recently received `action` is set.

        Input `tick` ensures that this node is I/O synchronized with the simulator."""
        # Set action in pybullet
        self.joint_cb(action.msgs[-1])
        # Send action that has been applied.
        return dict(action_applied=action.msgs[-1])

    @staticmethod
    def _joint_control(p, mode, bodyUniqueId, jointIndices, pos_gain, vel_gain, vel_target, max_vel, max_force):
        if mode == "position_control":

            def cb(action):
                for idx, a, v, kp, kd, mv, mf in zip(jointIndices, action, vel_target, pos_gain, vel_gain, max_vel, max_force):
                    p.setJointMotorControl2(
                        bodyIndex=bodyUniqueId,
                        jointIndex=idx,
                        controlMode=pybullet.POSITION_CONTROL,
                        targetPosition=a,
                        targetVelocity=v,
                        positionGain=kp,
                        velocityGain=kd,
                        maxVelocity=mv,
                        force=mf,
                        physicsClientId=p._client,
                    )
                return

        elif mode == "velocity_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.VELOCITY_CONTROL,
                    targetVelocities=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    forces=max_force,
                    physicsClientId=p._client,
                )

        elif mode == "pd_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.PD_CONTROL,
                    targetVelocities=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    forces=max_force,
                    physicsClientId=p._client,
                )

        elif mode == "torque_control":

            def cb(action):
                return p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.TORQUE_CONTROL,
                    forces=action,
                    positionGains=pos_gain,
                    velocityGains=vel_gain,
                    physicsClientId=p._client,
                )

        else:
            raise ValueError(f"Mode '{mode}' not recognized.")
        return cb
