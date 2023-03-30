import eagerx
from typing import Any, Dict
from eagerx.core.specs import EngineStateSpec
import pybullet


class PbXseriesGripper(eagerx.EngineState):
    @classmethod
    def make(cls, joints, constant, scale, fixed=False) -> EngineStateSpec:
        spec = cls.get_specification()
        spec.config.update(joints=joints, constant=constant, scale=scale, fixed=fixed)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Any):
        self.joints = spec.config.joints
        self.constant = spec.config.constant
        self.scale = spec.config.scale
        self.fixed = spec.config.fixed
        self.robot = simulator["object"]
        self._p = simulator["client"]
        self.physics_client_id = self._p._client

        self.bodyUniqueId = []
        self.jointIndices = []
        for _idx, pb_name in enumerate(spec.config.joints):
            bodyid, jointindex = self.robot.jdict[pb_name].get_bodyid_jointindex()
            self.bodyUniqueId.append(bodyid), self.jointIndices.append(jointindex)
        self.gripper_cb = self._gripper_reset(
            self._p, self.bodyUniqueId[0], self.jointIndices, self.constant, self.scale, self.fixed
        )

    def reset(self, state: Any):
        self.gripper_cb(state)

    @staticmethod
    def _gripper_reset(p, bodyUniqueId, jointIndices, constant, scale, fixed):
        def cb(state):
            # Mirror & scale gripper position
            pos = scale * state[0] + constant
            gripper_pos = [pos, -pos]

            # Only 1-dof joints are supported here.
            # https://github.com/bulletphysics/bullet3/issues/2803
            velocities = []
            states = p.getJointStates(bodyUniqueId=bodyUniqueId, jointIndices=jointIndices, physicsClientId=p._client)
            for _i, (_, vel, _, _) in enumerate(states):
                velocities.append([vel])
            p.resetJointStatesMultiDof(
                targetValues=[[pos] for pos in gripper_pos],
                targetVelocities=velocities,
                bodyUniqueId=bodyUniqueId,
                jointIndices=jointIndices,
                physicsClientId=p._client,
            )
            # If we are not performing control with the gripper, fix the position.
            if fixed:
                p.setJointMotorControlArray(
                    bodyUniqueId=bodyUniqueId,
                    jointIndices=jointIndices,
                    controlMode=pybullet.VELOCITY_CONTROL,
                    forces=len(jointIndices) * [10**9],
                    physicsClientId=p._client,
                )

        return cb


class LinkColorState(eagerx.EngineState):
    @classmethod
    def make(cls) -> EngineStateSpec:
        """A spec to create an EngineState that resets a specified link to the desired color.
        :return: EngineStateSpec
        """
        spec = cls.get_specification()
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Dict):
        """Initializes the engine state according to the spec."""
        self.robot = simulator["object"]
        self._p = simulator["client"]
        self.physics_client_id = self._p._client
        self.bodyUniqueId = self.robot.robot_objectid
        self.base_cb = self._link_color_reset(self._p, self.robot.parts, self.bodyUniqueId[0])

    def reset(self, state):
        """Resets the link state to the desired value."""
        self.base_cb(state.tolist())

    @staticmethod
    def _link_color_reset(p, parts, bodyUniqueId):
        def cb(state):
            for _pb_name, part in parts.items():
                bodyid, linkindex = part.get_bodyid_linkindex()
                p.changeVisualShape(bodyUniqueId, linkIndex=linkindex, rgbaColor=state)

        return cb


class TextureState(eagerx.EngineState):
    @classmethod
    def make(cls, texture_path) -> EngineStateSpec:
        """A spec to create an EngineState that resets a specified link to the desired color.
        :return: EngineStateSpec
        """
        spec = cls.get_specification()
        spec.config.update(texture_path=texture_path)
        return spec

    def initialize(self, spec: EngineStateSpec, simulator: Dict):
        """Initializes the engine state according to the spec."""
        self.robot = simulator["object"]
        p = simulator["client"]
        x = p.loadTexture(spec.config.texture_path)
        for _pb_name, part in self.robot.parts.items():
            bodyid, linkindex = part.get_bodyid_linkindex()
            p.changeVisualShape(self.robot.robot_objectid[0], linkIndex=linkindex, textureUniqueId=x)

    def reset(self, state):
        """Resets the link state to the desired value."""
        pass
