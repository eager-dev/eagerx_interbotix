from eagerx.core.entities import EngineState
import eagerx.core.register as register


class PbXseriesGripper(EngineState):
    @staticmethod
    @register.spec("PbXseriesGripper", EngineState)
    def spec(spec, joints, constant, scale):
        spec.initialize(PbXseriesGripper)
        spec.config.joints = joints
        spec.config.constant = constant
        spec.config.scale = scale

    def initialize(self, joints, constant, scale):
        self.obj_name = self.config["name"]
        flag = self.obj_name in self.simulator["robots"]
        assert flag, f'Simulator object "{self.simulator}" is not compatible with this simulation state.'
        self.joints = joints
        self.constant = constant
        self.scale = scale
        self.robot = self.simulator["robots"][self.obj_name]
        self._p = self.simulator["client"]
        self.physics_client_id = self._p._client

        self.bodyUniqueId = []
        self.jointIndices = []
        for _idx, pb_name in enumerate(joints):
            bodyid, jointindex = self.robot.jdict[pb_name].get_bodyid_jointindex()
            self.bodyUniqueId.append(bodyid), self.jointIndices.append(jointindex)
        self.gripper_cb = self._gripper_reset(self._p, self.bodyUniqueId[0], self.jointIndices, constant, scale)

    def reset(self, state, done):
        if not done:
            self.gripper_cb(state.data)

    @staticmethod
    def _gripper_reset(p, bodyUniqueId, jointIndices, constant, scale):
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

        return cb
