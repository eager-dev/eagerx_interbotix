from typing import Optional, List, Dict
from collections import deque
import numpy as np

import eagerx
from eagerx import Space
from eagerx.core.specs import NodeSpec
import eagerx.core.register as register
from eagerx.utils.utils import Msg, load
from eagerx_interbotix.safety import collision as col

# ROS imports
from urdf_parser_py.urdf import URDF


class SafePositionControl(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[int],
        upper: List[float],
        lower: List[float],
        vel_limit: List[float],
        duration: Optional[float] = None,
        checks: int = 2,
        collision: Dict = None,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> NodeSpec:
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param joints: joint names
        :param upper: upper joint limits
        :param lower: lower joint limits
        :param vel_limit: absolute velocity joint limits
        :param duration: time (seconds) it takes to reach the commanded positions from the current position.
        :param checks: collision checks performed over the duration.
        :param collision: A dict with the robot & workspace specification.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["goal", "current"]
        spec.config.outputs = ["filtered", "in_collision"]

        # Modify custom node params
        spec.config.joints = joints
        spec.config.upper = upper
        spec.config.lower = lower
        spec.config.vel_limit = vel_limit
        spec.config.duration = duration if isinstance(duration, float) else 2.0 / rate
        spec.config.checks = checks

        # Collision detector
        spec.config.collision = collision if isinstance(collision, dict) else None

        # Add converter & space
        spec.inputs.goal.space.update(low=lower, high=upper)
        spec.inputs.current.space.update(low=lower, high=upper)
        spec.outputs.filtered.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: NodeSpec):
        self.joints = spec.config.joints
        self.upper = np.array(spec.config.upper, dtype="float")
        self.lower = np.array(spec.config.lower, dtype="float")
        self.vel_limit = np.array(spec.config.vel_limit, dtype="float")
        self.duration = spec.config.duration
        self.checks = spec.config.checks
        self.dt = 1 / self.rate

        # Setup collision detector
        self.collision = spec.config.collision
        if self.collision is not None:
            self.collision_check = True
            # Setup physics server for collision checking
            import pybullet

            if self.collision.get("gui", False):
                self.col_id = pybullet.connect(pybullet.GUI)
            else:
                self.col_id = pybullet.connect(pybullet.DIRECT)
            # Load workspace
            bodies = load(self.collision["workspace"])(self.col_id)
            # Generate robot urdf (if not a path but a text file)
            r = self.collision["robot"]
            if r["urdf"].endswith(".urdf"):  # Full path specified
                fileName = r["urdf"]
            else:  # First write to /tmp file (else pybullet cannot load urdf)
                import uuid  # Use this to generate a unique filename

                fileName = f"/tmp/{str(uuid.uuid4())}.urdf"
                with open(fileName, "w") as file:
                    file.write(r["urdf"])
            # Load robot
            bodies["robot"] = pybullet.loadURDF(
                fileName,
                basePosition=r.get("basePosition", None),
                baseOrientation=r.get("baseOrientation", None),
                useFixedBase=r.get("useFixedBase", True),
                flags=r.get("flags", 0),
                physicsClientId=self.col_id,
            )
            urdf = URDF.from_xml_file(fileName)
            # Determine collision pairs
            self_collision_pairs, workspace_pairs = col.get_named_collision_pairs(bodies, urdf, self.joints)
            # Create collision detector
            self.self_collision = col.CollisionDetector(self.col_id, bodies, self.joints, self_collision_pairs)
            self.workspace = col.CollisionDetector(self.col_id, bodies, self.joints, workspace_pairs)
            # Set distance at which objects are considered in collision.
            self.margin = self.collision.get("margin", 0.0)
            # self._test_collision_tester(joints)
        else:
            self.collision_check = False

    def _test_collision_tester(self, joints):
        while True:
            # compute shortest distances for a random configuration
            q = np.pi * (np.random.random(len(joints)) - 0.5)
            in_col = self.self_collision.in_collision(q)

            print(f"[self_collision] In collision = {in_col}")

            in_col = self.workspace.in_collision(q)

            print(f"[workspace] In collision = {in_col}")

            # wait for user to press enter to continue
            input("Press <enter>!")

    @register.states()
    def reset(self):
        self.safe_poses = deque(maxlen=10)
        self.consecutive_unsafe = 0

    @register.inputs(goal=Space(dtype="float32"), current=Space(dtype="float32"))
    @register.outputs(filtered=Space(dtype="float32"), in_collision=Space(low=0, high=2, shape=(), dtype="int64"))
    def callback(self, t_n: float, goal: Msg = None, current: Msg = None):
        goal = goal.msgs[-1]
        current = current.msgs[-1]

        # Setpoint last safe position
        if self.collision_check:
            if (
                self.self_collision.in_collision(q=current, margin=self.margin)
                and self.self_collision.get_distance().min() < 0
            ):
                # self.backend.loginfo(f"[self_collision]: margin = {self.margin} | ds = {self.self_collision.get_distance().min()}")
                in_collision = np.array(1, dtype="int64")
            elif self.workspace.in_collision(margin=self.margin) and self.workspace.get_distance().min() < 0:
                # self.backend.loginfo(f"[workspace]: margin = {self.margin} | ds = {self.workspace.get_distance().min()}")
                in_collision = np.array(2, dtype="int64")
            else:
                in_collision = np.array(0, dtype="int64")
                if self.consecutive_unsafe == 0:
                    self.safe_poses.append(current)
        else:
            in_collision = np.array(0, dtype="int64")

        # Clip to joint limits
        filtered = np.clip(goal, self.lower, self.upper, dtype="float32")
        # Reduce goal to vel_limit
        diff = filtered - current
        too_fast = np.abs(diff / (self.duration)) > self.vel_limit
        if np.any(too_fast):
            filtered[too_fast] = current[too_fast] + np.sign(diff[too_fast]) * self.dt * self.vel_limit[too_fast]

        if self.collision_check:
            # Linearly interpolate for intermediate joint configurations
            t = np.linspace(self.dt / self.checks, self.dt, self.checks)
            interp = np.empty((current.shape[0], self.checks), dtype="float32")
            diff = filtered - current

            for i in range(current.shape[0]):
                interp[i][:] = np.interp(t, [0, self.duration], [current[i], current[i] + diff[i] * 1.02])

            for i in range(self.checks):
                if self.self_collision.in_collision(q=interp[:, i], margin=self.margin):
                    filtered = self.safe_poses[-1] if len(self.safe_poses) > 0 else current
                    self.consecutive_unsafe += 1
                    break
                elif self.workspace.in_collision(margin=self.margin):
                    filtered = self.safe_poses[-1] if len(self.safe_poses) > 0 else current
                    self.consecutive_unsafe += 1
                    break
                if i + 1 == self.checks:  # If all checks were successful (i.e. we did not break for-loop).
                    self.consecutive_unsafe = 0
        else:
            self.consecutive_unsafe = 0
            in_collision = np.array(0, dtype="int64")
        return dict(filtered=filtered, in_collision=in_collision)


class SafeVelocityControl(eagerx.Node):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        joints: List[int],
        upper: List[float],
        lower: List[float],
        vel_limit: List[float],
        duration: Optional[float] = None,
        checks: int = 2,
        collision: Dict = None,
        process: int = eagerx.NEW_PROCESS,
        color: str = "grey",
    ) -> NodeSpec:
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param name: Node name
        :param rate: Rate at which callback is called.
        :param joints: joint names
        :param upper: upper joint limits
        :param lower: lower joint limits
        :param vel_limit: absolute velocity joint limits
        :param duration: time (seconds) it takes to reach the commanded positions from the current position.
        :param checks: collision checks performed over the duration.
        :param collision: A dict with the robot & workspace specification.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: ENGINE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return: Node specification.
        """
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["goal", "position", "velocity"]
        spec.config.outputs = ["filtered", "in_collision"]

        # Modify custom node params
        spec.config.joints = joints
        spec.config.upper = upper
        spec.config.lower = lower
        spec.config.vel_limit = vel_limit
        spec.config.duration = duration if isinstance(duration, float) else 2.0 / rate
        spec.config.checks = checks

        # Collision detector
        spec.config.collision = collision if isinstance(collision, dict) else None

        # Set variable spaces
        spec.inputs.goal.space.update(low=[-v for v in vel_limit], high=vel_limit)
        spec.inputs.position.space.update(low=lower, high=upper)
        spec.outputs.filtered.space.update(low=lower, high=upper)
        return spec

    def initialize(self, spec: NodeSpec):
        self.joints = spec.config.joints
        self.upper = np.array(spec.config.upper, dtype="float")
        self.lower = np.array(spec.config.lower, dtype="float")
        self.vel_limit = np.array(spec.config.vel_limit, dtype="float")
        self.duration = spec.config.duration
        self.checks = spec.config.checks
        self.dt = 1 / self.rate

        # Setup collision detector
        self.collision = spec.config.collision
        if self.collision is not None:
            self.collision_check = True
            # Setup physics server for collision checking
            import pybullet

            if self.collision.get("gui", False):
                self.col_id = pybullet.connect(pybullet.GUI)
            else:
                self.col_id = pybullet.connect(pybullet.DIRECT)
            # Load workspace
            bodies = load(self.collision["workspace"])(self.col_id)
            # Generate robot urdf (if not a path but a text file)
            r = self.collision["robot"]
            if r["urdf"].endswith(".urdf"):  # Full path specified
                fileName = r["urdf"]
            else:  # First write to /tmp file (else pybullet cannot load urdf)
                import uuid  # Use this to generate a unique filename

                fileName = f"/tmp/{str(uuid.uuid4())}.urdf"
                with open(fileName, "w") as file:
                    file.write(r["urdf"])
            # Load robot
            bodies["robot"] = pybullet.loadURDF(
                fileName,
                basePosition=r.get("basePosition", None),
                baseOrientation=r.get("baseOrientation", None),
                useFixedBase=r.get("useFixedBase", True),
                flags=r.get("flags", 0),
                physicsClientId=self.col_id,
            )
            urdf = URDF.from_xml_file(fileName)
            # Determine collision pairs
            self_collision_pairs, workspace_pairs = col.get_named_collision_pairs(bodies, urdf, self.joints)
            # Create collision detector
            self.self_collision = col.CollisionDetector(self.col_id, bodies, self.joints, self_collision_pairs)
            self.workspace = col.CollisionDetector(self.col_id, bodies, self.joints, workspace_pairs)
            # Set distance at which objects are considered in collision.
            self.margin = self.collision.get("margin", 0.0)
            # self._test_collision_tester(joints)
        else:
            self.collision_check = False

    def _test_collision_tester(self, joints):
        while True:
            # compute shortest distances for a random configuration
            q = np.pi * (np.random.random(len(joints)) - 0.5)
            in_col = self.self_collision.in_collision(q)

            print(f"[self_collision] In collision = {in_col}")

            in_col = self.workspace.in_collision(q)

            print(f"[workspace] In collision = {in_col}")

            # wait for user to press enter to continue
            input("Press <enter>!")

    @register.states()
    def reset(self):
        self.safe_poses = deque(maxlen=10)
        self.consecutive_unsafe = 0

    @register.inputs(goal=Space(dtype="float32"), position=Space(dtype="float32"), velocity=Space(dtype="float32"))
    @register.outputs(filtered=Space(dtype="float32"), in_collision=Space(low=0, high=2, shape=(), dtype="int64"))
    def callback(self, t_n: float, goal: Msg = None, position: Msg = None, velocity: Msg = None):
        goal = goal.msgs[-1]
        position = position.msgs[-1]
        velocity = velocity.msgs[-1]

        # Setpoint last safe position
        if self.collision_check:
            if (
                self.self_collision.in_collision(q=position, margin=self.margin)
                and self.self_collision.get_distance().min() < 0
            ):
                in_collision = np.array(1, dtype="int64")
            elif self.workspace.in_collision(margin=self.margin) and self.workspace.get_distance().min() < 0:
                in_collision = np.array(2, dtype="int64")
            else:
                in_collision = np.array(0, dtype="int64")
                self.consecutive_unsafe = 0
                self.safe_poses.append(position)
        else:
            in_collision = np.array(0, dtype="int64")

        # Clip to velocity limits
        filtered = np.clip(goal, -self.vel_limit, self.vel_limit, dtype="float32")

        # Clip to joint limits
        max_position = position + filtered * self.dt
        clip_position = np.clip(max_position, self.lower, self.upper, dtype="float32")
        filtered = (clip_position - position) / self.dt

        if self.collision_check:
            # Linearly interpolate for intermediate joint configurations

            lin_vel = np.linspace(velocity, filtered, self.checks)
            interp = np.empty((position.shape[0], self.checks + 1), dtype="float32")

            interp[:, 0] = position
            interp[:, -1] = clip_position
            for i in range(self.checks):
                interp[:, i] += lin_vel[i] * self.dt / self.checks
                if i + 1 < self.checks:
                    interp[:, i + 1] = interp[:, i]

            for i in range(self.checks + 1):
                flag = self.self_collision.in_collision(q=interp[:, i], margin=self.margin)
                flag = flag | self.workspace.in_collision(margin=self.margin)
                if flag:
                    self.consecutive_unsafe += 1
                    if len(self.safe_poses) > 0:
                        idx = min(len(self.safe_poses), self.consecutive_unsafe + 1)
                        filtered = (self.safe_poses[-idx] - position) / self.dt
                        filtered = 0.5 * np.clip(filtered, -self.vel_limit, self.vel_limit, dtype="float32")
                    else:
                        filtered * 0
                    break
                elif i + 1 == self.checks:
                    self.consecutive_unsafe = 0
        else:
            self.consecutive_unsafe = 0
            in_collision = np.array(0, dtype="int64")
        return dict(filtered=filtered, in_collision=in_collision)
