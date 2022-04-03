from typing import Optional, List

# IMPORT ROS
from std_msgs.msg import UInt64, Float32MultiArray

# IMPORT EAGERX
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register

# IMPORT INTERBOTIX
from interbotix_xs_modules import arm


class XseriesSensor(EngineNode):
    @staticmethod
    @register.spec("XseriesSensor", EngineNode)
    def spec(
        spec,
        name: str,
        rate: float,
        joints: List[str],
        process: Optional[int] = process.BRIDGE,
        color: Optional[str] = "cyan",
        mode: str = "position",
    ):
        """XseriesSensor spec"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(XseriesSensor)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["obs"]

        # Set parameters, defined by the signature of cls.initialize(...)
        spec.config.joints = joints
        spec.config.mode = mode

    def initialize(self, joints, mode):
        self.obj_name = self.config["name"]
        self.joints = joints
        self.mode = mode
        self.robot = arm.InterbotixArmXSInterface(
            robot_model=self.config["robot_type"],
            group_name="arm",
            gripper_name="gripper",
            robot_name=self.obj_name,
            init_node=False
        )
        self.joint_cb = self._joint_measurement(self._p, self.mode, self.bodyUniqueId[0], self.jointIndices)

    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=UInt64)
    @register.outputs(obs=Float32MultiArray)
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        obs = self.joint_cb()
        return dict(obs=Float32MultiArray(data=obs))

    @staticmethod
    def _joint_measurement(p, mode, bodyUniqueId, jointIndices):
        def cb():
            states = p.getJointStates(bodyUniqueId=bodyUniqueId, jointIndices=jointIndices, physicsClientId=p._client)
            obs = []
            if mode == "position":  # (x)
                for _i, (pos, _vel, _force_torque, _applied_torque) in enumerate(states):
                    obs.append(pos)
            elif mode == "velocity":  # (v)
                for _i, (_pos, vel, _force_torque, _applied_torque) in enumerate(states):
                    obs.append(vel)
            elif mode == "force_torque":  # (Fx, Fy, Fz, Mx, My, Mz)
                for _i, (_pos, _vel, force_torque, _applied_torque) in enumerate(states):
                    obs += list(force_torque)
            elif mode == "applied_torque":  # (T)
                for _i, (_pos, _vel, _force_torque, applied_torque) in enumerate(states):
                    obs.append(applied_torque)
            else:
                raise ValueError(f"Mode '{mode}' not recognized.")
            return obs

        return cb