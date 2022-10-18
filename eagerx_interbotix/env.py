from scipy.spatial.transform import Rotation as R
import typing as t
import eagerx
import numpy as np
import gym


# Define environment
class ArmEnv(eagerx.BaseEnv):
    def __init__(self, name, rate, graph, engine, backend, max_steps: int, add_bias: bool = False, exclude_z: bool = True):
        super().__init__(name, rate, graph, engine, backend=backend, force_start=False)
        self.steps = 0
        self.max_steps = max_steps

        # Exclude
        self._exclude_z = exclude_z
        self._exclude_list = ["pos", "pos_desired"]

        # Bias
        self._add_bias = add_bias
        self.solid_bias = None
        self.yaw_bias = None

        # Rwd publishers
        self._pub_rwd = self.backend.Publisher(f"{self.ns}/environment/reward", "float32")

    @property
    def observation_space(self) -> gym.spaces.Space:
        obs_space = self._observation_space
        if self._exclude_z:
            names = obs_space.keys()
            for o in self._exclude_list:
                if o in names:
                    low, high = obs_space[o].low[:, :2], obs_space[o].high[:, :2]
                    obs_space[o] = gym.spaces.Box(low=low, high=high, dtype="float32")
        return obs_space

    def _exclude_obs(self, obs):
        names = obs.keys()
        for o in self._exclude_list:
            if o in names:
                obs[o] = obs[o][:, :2]
        return obs

    def step(self, action):
        # Step the environment
        self.steps += 1
        info = dict()
        obs = self._step(action)

        # Calculate reward
        yaw = obs["yaw"][0]
        yaw_des = obs["yaw_desired"][0]
        force = obs["force_torque"][0] if len(obs["force_torque"][0]) > 0 else 3 * [0.0]
        ee_pos = obs["ee_position"][0]
        goal_pos = obs["pos_desired"][0]
        achieved_pos = obs["pos"][0]
        vel = obs["velocity"][0] if "velocity" in obs else np.array([0], dtype="float32")
        des_vel = action["velocity"][0] if "velocity" in action else 0 * vel
        # Penalize distance of the end-effector to the object
        rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - obs["pos"][0]) - 0.05)
        # Penalize distance of the object to the goal
        yaw_error = min(abs(yaw_des - yaw), abs(yaw_des - 0.5 * np.pi - yaw), abs(yaw_des + 0.5 * np.pi - yaw))
        pos_error = np.linalg.norm(goal_pos - achieved_pos)
        rwd_pos = -4.0 * pos_error
        rwd_or = -((yaw_error / (1.0 + 10.0 * pos_error)) ** 2)
        # Penalize actions (indirectly, by punishing the angular velocity.
        rwd_ctrl = 0.1 * -np.linalg.norm(des_vel - vel)
        # Penalize force applied to box in vertical direction
        rwd_force = -0.0001 * (force[1] - 3.2) ** 2
        rwd = rwd_pos + rwd_ctrl + rwd_near + rwd_force + rwd_or
        # Print rwd build-up
        out_of_reach = np.linalg.norm(achieved_pos[:2]) > 1.0
        if out_of_reach:
            rwd = -50

        timed_out = self.steps >= self.max_steps
        if timed_out:
            info["TimeLimit.truncated"] = True

        done = out_of_reach | timed_out

        # Simulate bias in observations.
        if self._add_bias:
            obs["pos"] += self.solid_bias
            obs["yaw"] = (obs["yaw"] + self.yaw_bias) % (np.pi / 2)

        # Exclude z observations
        if self._exclude_z:
            obs = self._exclude_obs(obs)

        # Publish reward
        self._pub_rwd.publish(np.array([rwd, rwd_pos, rwd_or, rwd_ctrl, rwd_force, rwd_near], dtype="float32"))

        return obs, rwd, done, info

    def reset(self, states: t.Optional[t.Dict[str, np.ndarray]] = None):
        # Reset steps counter
        self.steps = 0

        # Sample bias
        self.solid_bias = 0.5 * 0.02 * (2 * np.random.random((3,)) - 1)
        self.yaw_bias = 0.5 * (0.1 * np.pi / 2) * (2 * np.random.random() - 1)

        # Sample states
        _states = self.state_space.sample()

        # Sample new starting orientation (vary yaw)
        yaw = np.random.random(()) * np.pi / 2
        self.state_space["solid/orientation"] = R.from_euler("zyx", [yaw, 0.0, 0.0]).as_quat().astype("float32")

        # Sample new starting state (at least 17 cm from goal)
        radius = 0.17
        while True:
            solid_pos = self.state_space["solid/position"].sample()
            goal_pos = self.state_space["goal/position"].sample()
            if np.linalg.norm(solid_pos[:2] - goal_pos[:2]) > radius:
                _states["solid/position"] = solid_pos
                _states["goal/position"] = goal_pos
                break

        # Set initial position state
        if "solid/aruco/position" in _states:
            self.state_space["solid/aruco/position"] = solid_pos

        # Overwrite with user provided states
        if states is not None:
            for key, value in states:
                if key in _states and value.shape == _states[key].shape:
                    _states[key] = value
                else:
                    self.backend.logwarn(f"State `{key}` incorrectly specified.")

        # Perform reset
        obs = self._reset(_states)

        # Simulate bias in observations.
        if self._add_bias:
            obs["pos"] += self.solid_bias
            obs["yaw"] = (obs["yaw"] + self.yaw_bias) % (np.pi / 2)

        # Exclude z observations
        if self._exclude_z:
            obs = self._exclude_obs(obs)
        return obs
