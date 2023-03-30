import eagerx
from eagerx.backends.single_process import SingleProcess
import eagerx_franka
import eagerx_interbotix

import yaml
from pathlib import Path
import numpy as np
import gym.wrappers as w


eagerx.set_log_level(eagerx.FATAL)
SingleProcess.MIN_THREADS = 100

root = Path(eagerx_interbotix.__file__).parent.parent
log_name = "HER_force_torque_2022-10-13-1836"
LOG_DIR = root / "logs" / f"{log_name}"


class VelocityControl:
    @staticmethod
    def make(name, rate, joint_names, joint_upper, joint_lower, vel_limit, checks=3, collision=None):
        from eagerx_utility.safety.node import SafeVelocityControl
        from eagerx_franka.franka_arm.franka_arm import FrankaArm

        if collision is None:
            robot_type = "panda"
            arm = FrankaArm.make(
                name=robot_type,
                robot_type=robot_type,
                sensors=["position", "velocity", "force_torque", "ee_pos", "ee_orn"],
                actuators=[],
                states=["position", "velocity", "gripper"],
                rate=1,
            )
            c = arm.config
            urdf = c.urdf
            module_path = Path(eagerx_franka.__file__).parent.parent / "assets"
            urdf_sbtd = urdf.replace("package://", str(module_path / "franka_panda") + "/")
            collision = dict(
                workspace="eagerx_utility.safety.workspaces/exclude_ground",
                margin=0.01,
                gui=False,
                robot=dict(urdf=urdf_sbtd, basePosition=c.base_pos, baseOrientation=c.base_or),
            )
        safe = SafeVelocityControl.make(
            name,
            rate,
            joint_names,
            joint_upper,
            joint_lower,
            [0.2 * vl for vl in vel_limit],
            checks=checks,
            collision=collision,
        )
        return safe


class Box:
    @staticmethod
    def make(name, rate, sensors):
        from eagerx_utility.solid.solid import Solid

        root = Path(eagerx_franka.__file__).parent.parent

        cam_path = root / "assets" / "calibrations"
        cam_name = "logitech_c170"
        with open(f"{cam_path}/{cam_name}.yaml", "r") as f:
            cam_intrinsics = yaml.safe_load(f)
        cam_translation = [1.5 * 0.811, 2 * 0.527, 2 * 0.43]
        cam_rotation = [0.321, 0.801, -0.466, -0.197]

        solid = Path(eagerx_franka.__file__).parent / "solid" / "assets"
        box_urdf = str(solid / "box.urdf")
        spec = Solid.make(
            name=name,
            rate=rate,
            sensors=sensors,
            urdf=box_urdf,
            states=["position", "velocity", "orientation", "angular_vel", "lateral_friction"],
            cam_translation=cam_translation,
            cam_rotation=cam_rotation,
            cam_index=2,
            cam_intrinsics=cam_intrinsics,
        )
        spec.sensors.position.space.update(low=[-1, -1, 0], high=[1, 1, 0.13])

        x, y, z = 0.45, 0.0, 0.05
        dx, dy = 0.1, 0.20
        spec.states.lateral_friction.space.update(low=0.1, high=0.4)
        spec.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
        spec.states.position.space.update(low=[x, -y - dy, z], high=[x + dx, y + dy, z])
        return spec


class BoxGoal:
    @staticmethod
    def make(name, rate, sensors):
        from eagerx_utility.solid.goal import Goal

        solid = Path(eagerx_franka.__file__).parent / "solid" / "assets"
        goal_urdf = str(solid / "box_goal.urdf")
        spec = Goal.make(
            name=name,
            urdf=goal_urdf,
            rate=rate,
            sensors=sensors,
            states=["position", "orientation"],
        )
        spec.sensors.position.space.update(low=[0, -1, 0], high=[1, 1, 0])

        x, y, z = 0.45, 0.0, 0.05
        dx, dy = 0.1, 0.20
        spec.states.orientation.space.update(low=[-1, -1, 0, 0], high=[1, 1, 0, 0])
        spec.states.position.space.update(low=[x, -y - dy, 0], high=[x + dx, y + dy, 0])
        return spec


class PandaArm:
    @staticmethod
    def make(name, sensors, actuators, rate):
        from eagerx_franka.franka_arm.franka_arm import FrankaArm

        robot_type = "panda"
        spec = FrankaArm.make(
            name=name,
            robot_type=robot_type,
            sensors=sensors,
            actuators=actuators,
            states=["position", "velocity", "gripper"],
            rate=rate,
        )
        spec.states.gripper.space.update(low=[0.0], high=[0.0])  # Set gripper to closed position
        spec.states.position.space.low[3] = -np.pi / 2
        spec.states.position.space.high[3] = -np.pi / 2
        spec.states.position.space.low[5] = np.pi / 2
        spec.states.position.space.high[5] = np.pi / 2
        return spec


class PandaIK:
    @staticmethod
    def make(name, rate):
        from eagerx_franka.ik.node import EndEffectorDownward
        import eagerx_franka.franka_arm.mr_descriptions as mrd

        robot_des = getattr(mrd, "panda")
        spec = EndEffectorDownward.make(
            name,
            rate,
            robot_des.Slist.tolist(),
            robot_des.M.tolist(),
            -np.pi,
            np.pi,
            max_dxyz=[0.2, 0.2, 0.2],
            max_dyaw=2 * np.pi / 2,
            eomg=0.01,
            ev=0.01,
        )
        return spec


class BoxPushEnv:
    @staticmethod
    def make(name, rate, graph, engine, backend):
        from eagerx_interbotix.env import ArmEnv
        from eagerx_interbotix.goal_env import GoalArmEnv

        T_max = 15.0  # [sec]
        env = ArmEnv(
            name=name,
            rate=rate,
            graph=graph,
            engine=engine,
            backend=backend,
            max_steps=int(T_max * rate),
            exclude_z=False,
        )
        goal_env = GoalArmEnv(env)
        env = w.rescale_action.RescaleAction(goal_env, min_action=-1.0, max_action=1.0)
        return env
