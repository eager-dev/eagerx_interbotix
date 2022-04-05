from typing import Optional, List, Dict

from collections import deque
from urdf_parser_py.urdf import URDF
import numpy as np

# IMPORT ROS
from std_msgs.msg import Float32MultiArray, UInt64

# IMPORT EAGERX
import eagerx.core.register as register
from eagerx.utils.utils import Msg, get_attribute_from_module
from eagerx.core.entities import Node, SpaceConverter
from eagerx.core.constants import process as p
from eagerx_interbotix.safety import collision as col


class SafetyFilter(Node):
    @staticmethod
    @register.spec("SafetyFilter", Node)
    def spec(
        spec,
        name: str,
        rate: float,
        joints: List[int],
        upper: List[float],
        lower: List[float],
        vel_limit: List[float],
        duration: Optional[float] = None,
        checks: int = 2,
        collision: Dict = None,
        process: Optional[int] = p.NEW_PROCESS,
        color: Optional[str] = "grey",
    ):
        """
        Filters goal joint positions that cause self-collisions or are below a certain height.
        Also check velocity limits.

        :param spec: Not provided by user.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :param joints: joint names
        :param upper: upper joint limits
        :param lower: lower joint limits
        :param vel_limit: absolute velocity joint limits
        :param duration: time (seconds) it takes to reach the commanded positions from the current position.
        :param checks: collision checks performed over the duration.
        :param collision: A dict with the robot & workspace specification.
        :param process: {0: NEW_PROCESS, 1: ENVIRONMENT, 2: BRIDGE, 3: EXTERNAL}
        :param color: console color of logged messages. {'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey'}
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(SafetyFilter)

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

        # Add converter & space_converter
        spec.inputs.goal.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32"
        )
        spec.inputs.current.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32"
        )
        spec.outputs.filtered.space_converter = SpaceConverter.make(
            "Space_Float32MultiArray", "$(config lower)", "$(config upper)", dtype="float32"
        )

    def initialize(
        self,
        joints: List[str],
        upper: List[float],
        lower: List[float],
        vel_limit: List[float],
        duration: float,
        checks: int,
        collision: dict,
    ):
        self.joints = joints
        self.upper = np.array(upper, dtype="float")
        self.lower = np.array(lower, dtype="float")
        self.vel_limit = np.array(vel_limit, dtype="float")
        self.duration = duration
        self.checks = checks
        self.dt = 1 / self.rate

        # Setup collision detector
        self.collision = collision
        if collision is not None:
            self.collision_check = True
            # Setup physics server for collision checking
            import pybullet

            if collision.get("gui", False):
                self.col_id = pybullet.connect(pybullet.GUI)
            else:
                self.col_id = pybullet.connect(pybullet.DIRECT)
            # Load workspace
            bodies = get_attribute_from_module(collision["workspace"])(self.col_id)
            # Generate robot urdf (if not a path but a text file)
            r = collision["robot"]
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
            self_collision_pairs, workspace_pairs = col.get_named_collision_pairs(bodies, urdf, joints)
            # Create collision detector
            self.self_collision = col.CollisionDetector(self.col_id, bodies, joints, self_collision_pairs)
            self.workspace = col.CollisionDetector(self.col_id, bodies, joints, workspace_pairs)
            # Set distance at which objects are considered in collision.
            # self.max_distance = collision.get("max_distance", 1.)
            self.margin = collision.get("margin", 0.0)
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

    @register.inputs(goal=Float32MultiArray, current=Float32MultiArray)
    @register.outputs(filtered=Float32MultiArray, in_collision=UInt64)
    def callback(self, t_n: float, goal: Msg = None, current: Msg = None):
        goal = np.array(goal.msgs[-1].data, dtype="float32")
        current = np.array(current.msgs[-1].data, dtype="float32")

        # Setpoint last safe position
        if self.collision_check:
            if (
                self.self_collision.in_collision(q=current, margin=self.margin)
                and self.self_collision.get_distance().min() < 0
            ):
                # rospy.loginfo(f"[self_collision]: margin = {self.margin} | ds = {self.self_collision.get_distance().min()}")
                in_collision = UInt64(data=1)
            elif self.workspace.in_collision(margin=self.margin) and self.workspace.get_distance().min() < 0:
                # rospy.loginfo(f"[workspace]: margin = {self.margin} | ds = {self.workspace.get_distance().min()}")
                in_collision = UInt64(data=2)
            else:
                in_collision = UInt64(data=0)
                if self.consecutive_unsafe == 0:
                    self.safe_poses.append(current)
        else:
            in_collision = UInt64(data=0)

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
                interp[i][:] = np.interp(t, [0, self.duration], [current[i], filtered[i] + diff[i] * 0.02])

            for i in range(self.checks):
                if self.self_collision.in_collision(q=interp[:, i], margin=self.margin):
                    filtered = self.safe_poses[-1] if len(self.safe_poses) > 0 else current
                    self.consecutive_unsafe += 1
                elif self.workspace.in_collision(margin=self.margin):
                    self.consecutive_unsafe += 1
                    filtered = self.safe_poses[-1] if len(self.safe_poses) > 0 else current
                else:
                    self.consecutive_unsafe = 0
        else:
            self.consecutive_unsafe = 0
            in_collision = UInt64(data=0)
        return dict(filtered=Float32MultiArray(data=filtered), in_collision=in_collision)
