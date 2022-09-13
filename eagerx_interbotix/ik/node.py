import typing as t
from scipy.spatial.transform import Rotation as R
import numpy as np
import modern_robotics as mr
import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg


class EndEffectorDownward(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: t.List[int],
        Slist: t.List[t.List[float]],
        M: t.List[t.List[float]],
        upper: t.List[float],
        lower: t.List[float],
        max_dxyz: t.List[float],
        max_dyaw: float,
        min_z: float = 0.03,
        eomg: float = 0.001,
        ev: float = 0.001,
        process: int = eagerx.NEW_PROCESS,
    ) -> NodeSpec:
        """
        Calculate desired joint positions based on delta pose of the end effector in task space using inverse kinematics.

        :param name: Node name.
        :param rate: Rate at which callback is called.
        :param joints: Joint names.
        :param Slist: The joint screw axes in the space frame when the manipulator is at the home position,
                      in the format of a matrix with axes as the columns. See modern robotics toolkit.
        :param M: The home configuration of the end-effector.
        :param upper: Upper joint limits.
        :param lower: Lower joint limits.
        :param max_dxyz: Maximum delta position in xyz [m/s].
        :param max_dyaw: Maximum delta rotation in yaw [rad/s].
        :param min_z: Minimum z position of the end-effector's link (cog?) [m].
        :param eomg: A small positive tolerance on the end-effector orientation
                     error. The returned joint angles must give an end-effector
                     orientation error less than eomg. See modern robotics toolkit.
        :param ev: A small positive tolerance on the end-effector linear position
                   error. The returned joint angles must give an end-effector
                   position error less than ev. See modern robotics toolkit.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}.
        :return: Parameter specification of the node.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["dxyz", "dyaw", "current", "xyz", "orn"]
        spec.config.outputs = ["target", "dtarget"]

        # Modify custom node params
        spec.config.joints = joints
        spec.config.Slist = Slist
        spec.config.M = M
        spec.config.max_dxyz = max_dxyz
        spec.config.max_dyaw = max_dyaw
        spec.config.eomg = eomg
        spec.config.ev = ev
        spec.config.min_z = min_z

        # Add converter & space
        spec.inputs.dxyz.space.update(low=[-i for i in max_dxyz], high=max_dxyz)
        spec.inputs.dyaw.space.update(low=-max_dyaw, high=max_dyaw)
        spec.inputs.current.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: NodeSpec):
        np.set_printoptions(precision=2, suppress=True)
        self.joints = spec.config.joints
        self.Slist = np.array(spec.config.Slist, dtype="float32")
        self.M = np.array(spec.config.M, dtype="float32")
        self.max_dxyz = np.array(spec.config.max_dxyz, dtype="float32")
        self.max_dyaw = np.array(spec.config.max_dyaw, dtype="float32")
        self.eomg = spec.config.eomg
        self.ev = spec.config.ev
        self.min_z = spec.config.min_z

    @register.states()
    def reset(self):
        pass

    @register.inputs(
        dxyz=Space(shape=(3,), dtype="float32"),
        xyz=Space(shape=(3,), dtype="float32"),
        orn=Space(shape=(4,), dtype="float32"),
        dyaw=Space(shape=(), dtype="float32"),
        current=Space(dtype="float32"),
    )
    @register.outputs(target=Space(dtype="float32"), dtarget=Space(dtype="float32"))
    def callback(self, t_n: float, dxyz: Msg, xyz: Msg, orn: Msg, dyaw: Msg, current: Msg):
        dxyz = dxyz.msgs[-1]
        xyz = xyz.msgs[-1]
        orn = orn.msgs[-1]
        dyaw = dyaw.msgs[-1]
        current = current.msgs[-1]

        # Limit dz
        dz = dxyz[-1]
        z = xyz[-1]
        dxyz[-1] = max(dz, self.max_dxyz[-1] * (-1 + 1 / np.exp(max(0, 10 * (z - self.min_z)))))

        # Calculate the target pose
        rot_ee2b = R.from_quat(orn).as_matrix()
        rot_red_ee2b = rot_ee2b[:2, 1:]
        yaw = np.arctan2(rot_red_ee2b[1, 1], rot_red_ee2b[0, 1])
        yaw_target = (yaw + dyaw / self.rate) % (2 * np.pi)
        yaw_target_cos = np.cos(yaw_target)
        yaw_target_sin = np.sin(yaw_target)
        rot_t2b = np.array([[0, -yaw_target_sin, yaw_target_cos], [0, yaw_target_cos, yaw_target_sin], [-1, 0, 0]])

        # 4x4 Transformation Matrix representing the transform from the
        # /<robot_name>/base_link frame to the /<robot_name>/ee_gripper_link frame
        T_sd = np.identity(4)
        T_sd[:3, :3] = rot_t2b
        T_sd[:3, 3] = xyz + dxyz / self.rate  # Scale delta position with rate

        theta_list, success = mr.IKinSpace(self.Slist, self.M, T_sd, current, self.eomg, self.ev)
        if success:
            target = np.array(theta_list, dtype="float32")
            dtarget = (target - current) * self.rate  # [rad/sec]
        else:
            # self.backend.logwarn("no solution")
            target = current
            dtarget = current * 0
        return dict(target=target, dtarget=dtarget)
