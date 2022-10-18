import gym
from gym import spaces
import numpy as np
from eagerx import BaseEnv


class GoalArmEnv(gym.Wrapper):
    def __init__(self, env: BaseEnv, add_bias: bool = False):
        super().__init__(env)
        self._add_bias = add_bias
        self._env = env
        flattened_space = spaces.Dict()
        space = dict(self._env.observation_space)
        space.pop("force_torque")
        self._og_des_pos_space = space.pop("pos_desired")
        flat_des_pos_space = spaces.flatten_space(self._og_des_pos_space)
        self._og_des_yaw_space = space.pop("yaw_desired")
        flat_des_yaw_space = spaces.flatten_space(self._og_des_yaw_space)
        self._og_desired_goal_space = spaces.Box(
            low=np.concatenate([flat_des_pos_space.low, flat_des_yaw_space.low]),
            high=np.concatenate([flat_des_pos_space.high, flat_des_yaw_space.high]),
        )

        self._og_observation_space = spaces.Dict(space)

        self._og_achieved_pos_space = space["pos"]
        flat_achieved_pos_space = spaces.flatten_space(self._og_achieved_pos_space)
        self._og_achieved_yaw_space = space["yaw"]
        flat_achieved_yaw_space = spaces.flatten_space(self._og_achieved_yaw_space)
        self._og_achieved_goal_space = spaces.Box(
            low=np.concatenate([flat_achieved_pos_space.low, flat_achieved_yaw_space.low]),
            high=np.concatenate([flat_achieved_pos_space.high, flat_achieved_yaw_space.high]),
        )

        flattened_space["desired_goal"] = spaces.flatten_space(self._og_desired_goal_space)
        flattened_space["observation"] = spaces.flatten_space(self._og_observation_space)
        flattened_space["achieved_goal"] = spaces.flatten_space(self._og_achieved_goal_space)
        self.observation_space = flattened_space
        self.action_space = spaces.flatten_space(self._env.action_space)

    def action(self, action: object) -> object:
        return spaces.unflatten(self._env.action_space, action)

    def observation(self, obs):
        goal_obs = dict()
        obs.pop("force_torque")
        flat_des_pos = spaces.flatten(self._og_des_pos_space, obs.pop("pos_desired"))
        flat_des_yaw = spaces.flatten(self._og_des_yaw_space, obs.pop("yaw_desired"))

        goal_obs["desired_goal"] = spaces.flatten(self._og_desired_goal_space, np.concatenate([flat_des_pos, flat_des_yaw]))

        flat_achieved_pos = spaces.flatten(self._og_achieved_pos_space, obs["pos"])
        flat_achieved_yaw = spaces.flatten(self._og_achieved_yaw_space, obs["yaw"])

        goal_obs["achieved_goal"] = spaces.flatten(
            self._og_achieved_goal_space, np.concatenate([flat_achieved_pos, flat_achieved_yaw])
        )
        # Simulate bias in observations.
        if self._add_bias:
            obs["pos"] = obs["pos"] + self.solid_bias
            obs["yaw"] = (obs["yaw"] + self.yaw_bias) % (np.pi / 2)

        goal_obs["observation"] = spaces.flatten(self._og_observation_space, obs)
        return goal_obs

    def step(self, action):
        action = dict(self.action(action))
        obs, rwd, done, info = self._env.step(action)

        # Calculate rewards independent of goal for HER
        force = obs["force_torque"][0] if len(obs["force_torque"][0]) > 0 else 3 * [0.0]
        ee_pos = obs["ee_position"][0]
        pos = obs["pos"][0]
        vel = obs["velocity"][0] if "velocity" in obs else np.array([0], dtype="float32")
        des_vel = action["velocity"][0] if "velocity" in action else 0 * vel
        rwd_near = 0.4 * -abs(np.linalg.norm(ee_pos - pos) - 0.05)
        rwd_ctrl = 0.1 * -np.linalg.norm(des_vel - vel)
        rwd_force = -0.0001 * (force[1] - 3.2) ** 2
        info["rwd_goal_independent"] = rwd_near + rwd_ctrl + rwd_force

        return self.observation(obs), rwd, done, info

    def reset(self):
        obs = self._env.reset()
        return self.observation(obs)

    def compute_reward(self, achieved_goal, desired_goal, info):
        pos_desired = desired_goal[:, :2]
        yaw_desired = desired_goal[:, -1]
        pos = achieved_goal[:, :2]
        yaw = achieved_goal[:, -1]

        # Penalize distance of the object to the pos_desired
        yaw_error = np.min(
            np.vstack(
                [np.abs(yaw_desired - yaw), np.abs(yaw_desired - 0.5 * np.pi - yaw), np.abs(yaw_desired + 0.5 * np.pi - yaw)]
            ),
            axis=0,
        )
        dist = pos_desired - pos
        pos_error = np.linalg.norm(dist.reshape(-1, 2), axis=1)
        rwd_dist = -4.0 * pos_error - (yaw_error / (1 + 10 * pos_error)) ** 2
        # Add goal independent rewards
        rwd_goal_independent = np.asarray([i["rwd_goal_independent"] for i in info])
        rwd = rwd_dist + rwd_goal_independent
        out_of_reach = np.linalg.norm(achieved_goal[:, :2], axis=1) > 1.0
        rwd[out_of_reach] = -50
        return rwd
