import eagerx
from typing import Any
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
