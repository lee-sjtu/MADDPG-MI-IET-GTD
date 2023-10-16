import numpy as np
from gym import spaces
import random

class MultiAgentEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 ag_observation_callback=None, st_observation_callback=None):

        self.world = world
        self.agents = self.world.agents
        self.n = len(world.agents)
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.ag_observation_callback = ag_observation_callback
        self.st_observation_callback = st_observation_callback
        self.action_space = []
        self.observation_space = []
        # self.b = 0.2
        self.alpha = 0.5
        for agent in self.agents:
            self.action_space.append(spaces.Box(low=-agent.p_range, high=+agent.p_range, shape=(1,), dtype=np.float32))
            # obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(7,), dtype=np.float32))

    def reset(self, num_EVs_per_agent):
        obs_N = []
        ag_obs_N = []
        st_obs = []
        soc_, e_up, e_down, t_stay_ = self.reset_callback(self.world, num_EVs_per_agent)
        # print(soc_, e_up, e_down, t_stay_)
        for i in range(num_EVs_per_agent):
            ag_obs_n = []
            for j, agent in enumerate(self.agents):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
            ag_obs_N.append(ag_obs_n)
        # for i, agent in enumerate(self.agents):
        st_obs, num_ev = self._get_st_obs()
        # print("1", ag_obs_N, st_obs)
        for i in range(num_EVs_per_agent):
            obs_n = []
            # print(np.shape(st_obs))
            for j, agent in enumerate(self.agents):
                # print("2", ag_obs_N[i][j])
                obs_n.append(np.concatenate((ag_obs_N[i][j], st_obs), axis=-1))
            obs_N.append(obs_n)
        return obs_N, soc_, e_up, e_down, t_stay_, num_ev

    def step(self, action_n, obs_n, num_EVs_per_agent, soc, e_up, e_down, t_stay, tot_step):
        ag_obs_N = []
        obs_N = []
        eme = 0
        reward_N = []
        # st_obs = []
        rp1_N = []
        self.agents = self.world.agents
        self.stations = self.world.stations

        p_n, soc_, t_stay_ = self.world.step(action_n, obs_n, soc, t_stay)   # agent.state/station.state更新

        reward_dim2 = self._get_reward(p_n)
        # print(reward_dim2)
        for i in range(num_EVs_per_agent):
            rp1_n = []
            for j, agent in enumerate(self.agents):
                rp1_n.append(reward_dim2[1])
            # print('rp1n', rp1_n)
            rp1_N.append(rp1_n)
        # print("rp1_N", rp1_N)
        rc_a_n = (reward_dim2[0])

        for i in range(num_EVs_per_agent):
            reward_n = []
            for j, agent in enumerate(self.agents):
                # print("3", i, j, rc_a_n[i][j])
                # r_tot = (1-self.alpha)*rc_a_n[i][j] / 6 + self.alpha*rp1_N[i][j]
                r_tot = (rc_a_n[i][j] / 6 + rp1_N[i][j])/2
                reward_n.append(r_tot)
            reward_N.append(reward_n)
        for i, agent in enumerate(self.agents):
            agent.state.eme = 0
            agent.state.n_ev = 0

        for station in self.stations:
            station.state.t_cur += 1
            station.state.n_ev = 0
            station.state.eme_tot = 0

        for i in range(num_EVs_per_agent):
            ag_obs_n = []
            for j, agent in enumerate(self.agents):
                ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
            ag_obs_N.append(ag_obs_n)
        st_obs, num_ev = self._get_st_obs()
        for i in range(num_EVs_per_agent):
            obs_n = []
            for j, agent in enumerate(self.agents):
                obs_n.append(np.concatenate((ag_obs_N[i][j], st_obs), axis=-1))
            obs_N.append(obs_n)
        # for i in range(num_EVs_per_agent):
        #     for j, agent in enumerate(self.agents):
        #         ag_obs_n.append(self._get_ag_obs(agent, soc_[i][j], e_up[i][j], e_down[i][j], t_stay_[i][j]))
        # for i, agent in enumerate(self.agents):
        #     st_obs.append(self._get_st_obs())
        # for i in range(num_EVs_per_agent):
        #     obs_n = []
        #     for j, agent in enumerate(self.agents):
        #         obs_n.append(np.concatenate((ag_obs_n[i*len(self.agents)+j], st_obs[j]), axis=-1))
        #     obs_N.append(obs_n)
        return obs_N, p_n, reward_N, rc_a_n, rp1_N, num_ev#, rp2_n

    # get observation for a particular agent
    def _get_ag_obs(self, agent, soc_, e_up, e_down, t_stay):
        return self.ag_observation_callback(agent, self.world, soc_, e_up, e_down, t_stay)

    # get observation for a particular agent
    def _get_st_obs(self):
        return self.st_observation_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self, p_n):
        return self.reward_callback(self.world, p_n)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        # print(action[0])
        agent.action.p = action[0]
